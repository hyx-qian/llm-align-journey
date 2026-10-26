"""Multi-turn tool-call rollout: generate -> parse <tool_call> -> execute -> feedback -> continue.

★★★ 核心逻辑（多轮循环 + 终止条件 + 最终答案提取）由你实现——
这是 Agent RL 的心脏：单轮 rollout（GRPO v2）到多轮工具调用 rollout 的关键升级。
骨架与提示已给全，按 TODO 顺序填写。
"""
import re

from tool_executor import safe_calc

TOOL_CALL_RE = re.compile(r"<tool_call>\s*calc\((.*?)\)\s*</tool_call>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
MAX_TURNS = 5


# ========== TODO 1（学员实现）：单轮生成封装 ==========
def generate_once(llm_client, messages: list, sampling_params) -> str:
    """调用 vLLM 生成一轮文本。

    Args:
        messages: list[dict]，累积对话（{"role": ..., "content": ...}），
                  含 system（工具说明）+ user（题目）+ 历史 assistant/tool 轮
        sampling_params: vLLM SamplingParams（temperature/top_p/max_tokens）
    Returns:
        str: 本轮 assistant 生成的文本
    提示：与 grpo_smoke_v2.py 的 rollout 写法对齐——
          vLLM chat 接口可直接吃 messages 列表；或 generate + 手动拼接模板。
    """
    # raise NotImplementedError("TODO 1(学员): 对接 vLLM 生成")
    return llm_client(messages, sampling_params)
    


def rollout_multiturn(llm_client, prompt_text: str, sampling_params, max_turns: int = MAX_TURNS) -> dict:
    """多轮工具调用 rollout。

    循环规则：
      每轮 generate_once 生成 assistant 文本 → append 轨迹；
      用 TOOL_CALL_RE 解析 <tool_call>calc(...)</tool_call>：
        - 有调用 → safe_calc(expr) 执行（try/except 包住，异常时工具返回错误提示）→
          tool 结果轮 append（role="tool"）→ num_tool_calls += 1 → 进入下一轮；
        - 无调用 → 终止；
      已出现 \\boxed{...} → 终止（拿到最终答案）；
      达到 max_turns → 终止。

    Returns:
        dict: {
            "trajectory": [(role, content), ...],   # 完整轨迹，assistant 与 tool 轮交替
            "final_answer": str | None,             # 从最后一个 \\boxed 提取
            "num_tool_calls": int,
            "turns_used": int,
        }
    """
    messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    trajectory = []
    num_tool_calls = 0
    final_answer = None
    rollout = {
        "trajectory": [],
        "final_answer": None,
        "num_tool_calls": 0,
        "turns_used": 0,
    }
        
    # raise NotImplementedError("TODO 2(学员): 多轮循环逻辑")
    
    for turn in range(max_turns):
        text = generate_once(llm_client, messages, sampling_params)
        trajectory.append(('assistant', text))
        tool_calls = TOOL_CALL_RE.findall(text)
        boxed = BOXED_RE.findall(text)
        if tool_calls:       # 有工具调用 → 执行 → 回填 → 继续
            messages.append({'role': 'assistant', 'content': text})
            for tool_call in tool_calls:
                try:
                    result = safe_calc(tool_call)
                    tool_msg = f'<tool_result>{result}</tool_result>'
                except Exception as e:
                    tool_msg = f"<tool_result>ERROR: {e}</tool_result>"
                trajectory.append(('tool', tool_msg))
                messages.append({'role': 'user', 'content': tool_msg})
                num_tool_calls += 1
        elif boxed:         # 已给最终答案 → 终止
            final_answer = boxed[-1]
            break
        else:               # 无工具调用且无最终答案 → 终止
            break
        
    rollout['trajectory'] = trajectory
    rollout['final_answer'] = final_answer
    rollout['num_tool_calls'] = num_tool_calls
    rollout['turns_used'] = sum(1 for role, content in trajectory if role == 'assistant')
    return rollout


TOOL_SYSTEM_PROMPT = (
    "You are a math assistant with access to a calculator tool.\n"
    "To use it, output exactly: <tool_call>calc(EXPRESSION)</tool_call>\n"
    "EXPRESSION may contain numbers, + - * / ** % and functions sqrt/abs/round/min/max.\n"
    "After you see <tool_result>...</tool_result>, continue reasoning.\n"
    "When you reach the final answer, output it inside \\boxed{...}."
)

if __name__ == "__main__":
    # 离线自测：不调 vLLM，用手写的假 assistant 输出验证解析/执行/终止逻辑
    class FakeLLM:
        def __init__(self):
            self.turn = 0
        def __call__(self, messages, sp):
            self.turn += 1
            if self.turn == 1:
                return "Let me compute 23*47: <tool_call>calc(23*47)</tool_call>"
            if self.turn == 2:
                return "23*47 = 1081. So the answer is \\boxed{1081}"
            return "done"
    r = rollout_multiturn(FakeLLM(), "What is 23*47?", sampling_params=None)
    assert r["final_answer"] == "1081", r
    assert r["num_tool_calls"] == 1
    print("FAKE ROLLOUT OK:", r["final_answer"], "| turns:", r["turns_used"])
