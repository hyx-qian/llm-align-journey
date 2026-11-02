# -*- coding: utf-8 -*-
"""SFT 冷启动数据构造：RFT CoT → 工具调用示范轨迹。

流程：RFT jsonl（prompt + completion）→ LaTeX 转纯文本 → 算式识别改写
  → safe_calc 重放验证（心算结果与真实计算一致性检查——轨迹正确性保证）
  → 质量过滤（≥1 个有效改写）→ 上限 200 条 → agent 格式输出

分工：Part 1/2（LaTeX 转换 + 算式改写）由你实现——这是数据构造的核心逻辑；
Part 3（流程/过滤/输出/验证框架）已写好。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_executor import safe_calc
from rollout_multiturn import TOOL_SYSTEM_PROMPT   # 统一简短版（与推理时一致，train-serve 一致性）

RFT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gsm8k_rft.jsonl")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_coldstart.jsonl")
MAX_SAMPLES = 200          # 格式学习不需要大数据量，上限防过拟合题目内容
MIN_REWRITES = 1           # 至少 1 个算式被改写才保留（纯逻辑推理题无示范价值）


# ==================== Part 1（已实现）：LaTeX → 纯文本（pylatexenc + 保护回填） ====================
from pylatexenc.latex2text import LatexNodes2Text

PLACEHOLDER_RE = re.compile(r"@@PROTECT(\d+)@@")
PROTECTED = [re.compile(r"\\boxed\{[^{}]*\}"), re.compile(r"\\text\{[^{}]*\}")]


def latex_to_plain(text: str) -> str | None:
    """三段式：保护 \\boxed    ext → pylatexenc 转换 → 回填 + 符号后处理。失败返回 None。"""
    try:
        # 1. 保护 pylatexenc 不认识的命令（boxed 会被剥成 boxed72、    ext 会被吃掉）
        protected = []
        def _protect(m):
            protected.append(m.group(0))
            return f"@@PROTECT{len(protected)-1}@@"
        for pat in PROTECTED:
            text = pat.sub(_protect, text)
        # 2. pylatexenc 结构转换（\frac→/、\[\] 剥壳、嵌套正确处理）
        plain = LatexNodes2Text(math_mode="text").latex_to_text(text)
        # 3. 回填
        plain = PLACEHOLDER_RE.sub(lambda m: protected[int(m.group(1))], plain)
        # 4. 后处理：Unicode 数学符号 → safe_calc 可解析（×÷− 是 pylatexenc 的输出）
        plain = plain.replace("\u00d7", "*").replace("\u00f7", "/").replace("\u2212", "-")
        plain = re.sub(r"\n\s*\n+", "\n\n", plain)   # 多余空行压缩
        return plain
    except Exception:
        return None   # 转换失败整条跳过（错误示范比没示范糟）


# ==================== Part 2（★学员实现）：算式识别与改写 ====================
NUM = r"-?\d+(?:\.\d+)?"          # 严格的数字：-12 或 3.14
OP  = r"\s*[+\-*/]\s*"
EXPRESSION_RE = re.compile(
    rf"({NUM}(?:{OP}{NUM})+)\s*=\s*({NUM})"
)

def rewrite_to_tool_calls(text: str) -> tuple[str, int] | None:
    """识别 `表达式 = 结果` 模式 → 改写成 tool_call/tool_result 对。

    TODO(学员-核心2)：
      1. EXPRESSION_RE 逐个匹配 `... = 结果` 模式
      2. 对每个匹配用 safe_calc(expr) 【重放验证】：
         - abs(safe_calc(expr) - float(result)) < 1e-6 → 心算正确 → 改写为
             <tool_call>calc(expr_clean)</tool_call>\n<tool_result>result</tool_result>
         - 不符 → 保留原文不改写（改写出错误轨迹比不改更糟——轨迹正确性保证）
         - safe_calc 本身抛异常（除零等）→ 同样保留原文
      3. 表达式清洗：去掉多余空格再放进 calc()（safe_calc 能解析的格式）
      4. 返回改写后的完整文本 + 改写次数
    Returns:
        (str, int) | None: (改写后文本, 改写次数)；无任何改写返回 None
    """
    matches = list(EXPRESSION_RE.finditer(text))
    if not matches:
        return None, 0

    out = text
    count = 0
    for m in reversed(matches):                      # 倒序：从最后一个匹配开始替换
        expr_raw, result = m.group(1), m.group(2)
        expr_clean = re.sub(r"\s+", "", expr_raw)    # 去空格 → safe_calc 可解析
        try:
            if abs(safe_calc(expr_clean) - float(result)) < 1e-6:
                repl = (f"<tool_call>calc({expr_clean})</tool_call>\n"
                        f"<tool_result>{result}</tool_result>")
                out = out[:m.start()] + repl + out[m.end():]   # span 切片替换
                count += 1
        except Exception:
            pass                                     # 重放失败 → 保留原文不改写
    return (out, count) if count > 0 else (None, 0)  # 无改写 → None（Part 3 过滤）
    # raise NotImplementedError("TODO(学员-核心2): 算式改写")


# ==================== Part 3（已写好）：流程 / 过滤 / 输出 ====================
def main():
    n_in = n_ok = n_skip = 0
    out_rows = []
    with open(RFT_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            n_in += 1
            comp = (ex.get("completion") or [{}])[0].get("content", "")
            if not comp:
                n_skip += 1
                continue
            plain = latex_to_plain(comp)
            if plain is None:
                n_skip += 1
                continue
            rewritten, n_rw = rewrite_to_tool_calls(plain)
            if rewritten is None or n_rw < MIN_REWRITES:
                n_skip += 1
                continue
            out_rows.append({
                "prompt": [
                    {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                    {"role": "user", "content": ex["prompt"][0]["content"] if isinstance(ex.get("prompt"), list) else ex["prompt"]},
                ],
                "completion": [{"role": "assistant", "content": rewritten}],
                "n_rewrites": n_rw,
                "source": "rft_coldstart",
            })
            n_ok += 1
            if n_ok >= MAX_SAMPLES:
                break

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_rw = sum(r["n_rewrites"] for r in out_rows) / max(1, len(out_rows))
    print(f"cold_start_data: 输入 {n_in} → 保留 {n_ok} / 跳过 {n_skip}（上限 {MAX_SAMPLES}）")
    print(f"平均改写次数: {avg_rw:.1f}  → {OUT_PATH}")
    # 抽 1 条打印人工检查
    if out_rows:
        print("=== 样例检查（第 1 条 completion）===")
        print(out_rows[0]["completion"][0]["content"][:500])


if __name__ == "__main__":
    main()
