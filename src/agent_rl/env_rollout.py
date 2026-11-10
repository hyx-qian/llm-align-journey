"""Multi-turn tool-call rollout: generate -> parse <tool_call> -> execute -> feedback -> continue.

支持三工具（T2/T3 演化实验）：
  calc(EXPR)                    单步算术（env_tool_calc）
  python("CODE")                代码执行 sandbox（env_tool_python）
  sympy("EXPR", "OP"[, "VAR"])  符号运算（env_tool_sympy）

设计要点：
  ① 三个专用正则（兼容 stop=["</tool_call>"] 的截断）
  ② find_tool_calls 做**区间去重**：长匹配（python/sympy）优先，落在已消费区间内的短匹配（如 python 参数字符串里的 calc 字样）丢弃
  ③ extract_boxed 用**平衡花括号扫描**（支持 \\boxed{\\frac{1}{2}}）
  ④ 轨迹保持 (role, content) 二元组；tool_usage / stop_reason 在返回值
  ⑤ system_prompt 可注入（训练回放 TOOL_SYSTEM_PROMPT；B/C 文档路线 TOOL_DOC_V1）
"""
import re
from collections import Counter

from env_tool_calc import safe_calc
from env_tool_python import safe_python
from env_tool_sympy import safe_sympy

# calc：无引号表达式；[^<>]+ 排尖括号（遇 </tool_call> 的 < 即停），兼容嵌套括号
CALC_RE = re.compile(r"<tool_call>\s*calc\(([^<>]+)\)", re.DOTALL)
# python：参数为字符串（单/双引号均可）
PYTHON_RE = re.compile(r"""<tool_call>\s*python\(\s*["'](.*?)["']\s*\)""", re.DOTALL)
# sympy：2-3 个字符串参数
SYMPY_RE = re.compile(
    r"""<tool_call>\s*sympy\(\s*["'](.*?)["']\s*,\s*["'](.*?)["']\s*(?:,\s*["'](.*?)["']\s*)?\)""",
    re.DOTALL,
)

# 兼容旧代码：TOOL_CALL_RE 仅匹配 calc（训练数据用）
TOOL_CALL_RE = CALC_RE
# 简单正则（不处理嵌套花括号；保留兼容，rollout 内部用 extract_boxed）
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
MAX_TURNS = 5


def extract_boxed(text: str) -> list:
    """提取所有 \\boxed{...}（平衡花括号扫描，支持 \\boxed{\\frac{1}{2}}）。"""
    out = []
    i = 0
    marker = "\\boxed{"
    while True:
        j = text.find(marker, i)
        if j < 0:
            break
        k = j + len(marker)
        depth = 1
        while k < len(text) and depth > 0:
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        if depth == 0:
            out.append(text[j + len(marker):k - 1])
            i = k
        else:  # 未闭合 → 停止
            break
    return out


def find_tool_calls(text: str):
    """返回 [(tool_name, args_tuple, position)]，按位置排序并做区间去重。

    去重规则：按 (start, -(end-start)) 排序（位置靠前且更长者优先），
    完全落在已消费区间内的小匹配被丢弃——避免 python 参数字符串里的 calc 被误执行。
    """
    raw = []
    for m in SYMPY_RE.finditer(text):
        raw.append(("sympy", (m.group(1), m.group(2), (m.group(3) or "x").strip() or "x"), m.start(), m.end()))
    for m in PYTHON_RE.finditer(text):
        raw.append(("python", (m.group(1),), m.start(), m.end()))
    for m in CALC_RE.finditer(text):
        raw.append(("calc", (m.group(1),), m.start(), m.end()))

    raw.sort(key=lambda x: (x[2], -(x[3] - x[2])))  # 位置靠前 + 更长优先
    consumed = []
    out = []
    for name, args, s, e in raw:
        if any(s >= cs and e <= ce for cs, ce in consumed):
            continue  # 落在已消费区间内 → 丢弃（如 python 参数串里的 calc 字样）
        out.append((name, args, s))
        consumed.append((s, e))
    return sorted(out, key=lambda x: x[2])


def execute_tool(name: str, args: tuple) -> str:
    """分发到对应工具；异常统一转 ERROR 字符串。"""
    try:
        if name == "calc":
            return safe_calc(args[0])
        if name == "python":
            return safe_python(args[0])
        if name == "sympy":
            return safe_sympy(args[0], args[1], args[2] if len(args) > 2 else "x")
        return f"ERROR: unknown tool '{name}'"
    except Exception as e:  # 兜底
        return f"ERROR: {type(e).__name__}: {e}"


def generate_once(llm_client, messages: list, sampling_params) -> str:
    """调用 vLLM 生成一轮文本。"""
    return llm_client(messages, sampling_params)


def rollout_multiturn(llm_client, prompt_text: str, sampling_params,
                      max_turns: int = MAX_TURNS, system_prompt: str | None = None) -> dict:
    """多轮工具调用 rollout。

    Returns:
        dict: trajectory / final_answer / num_tool_calls / turns_used / tool_usage / stop_reason
    """
    messages = [
        {"role": "system", "content": system_prompt if system_prompt is not None else TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    trajectory = []
    num_tool_calls = 0
    final_answer = None
    tool_usage = Counter()
    stop_reason = "no_output"

    for _turn in range(max_turns):
        text = generate_once(llm_client, messages, sampling_params)
        trajectory.append(("assistant", text))

        boxed = extract_boxed(text)
        if boxed:  # 已给最终答案 → 终止
            final_answer = boxed[-1]
            stop_reason = "boxed"
            break

        calls = find_tool_calls(text)
        if calls:  # 有调用 → 逐个执行 → 回填 → 继续
            messages.append({"role": "assistant", "content": text})
            for name, args, _pos in calls:
                result = execute_tool(name, args)
                tool_msg = f"<tool_result>{result}</tool_result>"
                trajectory.append(("tool", tool_msg))
                messages.append({"role": "user", "content": tool_msg})
                num_tool_calls += 1
                tool_usage[name] += 1
            stop_reason = "tool_call"
        else:
            stop_reason = "no_output"
            break

    if final_answer is None and stop_reason == "tool_call" and num_tool_calls > 0:
        if sum(1 for r, _ in trajectory if r == "assistant") >= max_turns:
            stop_reason = "max_turns"

    return {
        "trajectory": trajectory,
        "final_answer": final_answer,
        "num_tool_calls": num_tool_calls,
        "turns_used": sum(1 for role, _ in trajectory if role == "assistant"),
        "tool_usage": dict(tool_usage),
        "stop_reason": stop_reason,
    }


TOOL_SYSTEM_PROMPT = (
    "You are a math assistant with access to a calculator tool.\n"
    "To use it, output exactly: <tool_call>calc(EXPRESSION)</tool_call>\n"
    "EXPRESSION may contain numbers, + - * / ** % and functions sqrt/abs/round/min/max.\n"
    "After you see <tool_result>...</tool_result>, continue reasoning.\n"
    "When you reach the final answer, output it inside \\boxed{...}."
)


if __name__ == "__main__":
    class FakeLLM:
        def __init__(self, script):
            self.script = list(script)
            self.i = 0

        def __call__(self, messages, sp):
            out = self.script[self.i] if self.i < len(self.script) else "done"
            self.i += 1
            return out

    # 1) calc 两轮
    r = rollout_multiturn(FakeLLM([
        "Let me compute 23*47: <tool_call>calc(23*47)</tool_call>",
        "23*47 = 1081. So the answer is \\boxed{1081}",
    ]), "What is 23*47?", None)
    assert r["final_answer"] == "1081" and r["num_tool_calls"] == 1 and r["tool_usage"] == {"calc": 1}, r
    print("OK calc:", r["final_answer"], r["tool_usage"], r["stop_reason"])

    # 2) python 一轮
    r = rollout_multiturn(FakeLLM([
        'I will loop: <tool_call>python("print(sum(range(1,11)))")</tool_call>',
        'The sum is 55. \\boxed{55}',
    ]), "sum 1..10", None)
    assert r["num_tool_calls"] == 1 and r["tool_usage"] == {"python": 1} and r["final_answer"] == "55", r
    print("OK python:", r["final_answer"], r["tool_usage"])

    # 3) sympy 两参数
    r = rollout_multiturn(FakeLLM([
        'Use sympy: <tool_call>sympy("3*x + 5 = 20", "solve")</tool_call>',
        'x = 5. \\boxed{5}',
    ]), "solve 3x+5=20", None)
    assert r["num_tool_calls"] == 1 and r["tool_usage"] == {"sympy": 1}, r
    print("OK sympy:", r["trajectory"][1][1][:30], r["tool_usage"])

    # 4) 触顶
    r = rollout_multiturn(FakeLLM(['<tool_call>calc(1+1)</tool_call>'] * 10), "q", None, max_turns=3)
    assert r["stop_reason"] == "max_turns" and r["num_tool_calls"] == 3, r
    print("OK max_turns:", r["num_tool_calls"], r["stop_reason"])

    # 5) 无输出
    r = rollout_multiturn(FakeLLM(["I do not know"]), "q", None)
    assert r["stop_reason"] == "no_output" and r["num_tool_calls"] == 0, r
    print("OK no_output:", r["stop_reason"])

    # 6) ★ 回归：一轮内两个 calc 调用（陷阱 ① 反向验证）
    calls = find_tool_calls('<tool_call>calc(1+1)</tool_call> and <tool_call>calc(2+2)</tool_call>')
    assert [c[1][0] for c in calls] == ["1+1", "2+2"], calls
    print("OK two-calc:", [c[1][0] for c in calls])

    # 7) ★ 回归：python 参数串里含 calc 字样（陷阱 ② 修复验证）
    t2 = """<tool_call>python("s='<tool_call>calc(1+1)</tool_call>'")</tool_call>"""
    calls = find_tool_calls(t2)
    assert len(calls) == 1 and calls[0][0] == "python", calls
    print("OK nested-dedup:", calls[0][0])

    # 8) ★ 回归：嵌套花括号 boxed（陷阱 ③ 修复验证）
    assert extract_boxed(r"\\boxed{\\frac{1}{2}}") == [r"\\frac{1}{2}"], extract_boxed(r"\\boxed{\\frac{1}{2}}")
    assert extract_boxed("\\boxed{72}") == ["72"]
    print("OK nested-boxed:", extract_boxed(r"\\boxed{\\frac{1}{2}}"))

    print("ALL ROLLOUT TESTS PASSED")
