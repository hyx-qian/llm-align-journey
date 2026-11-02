# -*- coding: utf-8 -*-
"""边界示范构造 + 冷启动 v2 数据混合（300 条）。

A 完成即停 15 条：completion 截断到最后一个 boxed 闭合处
B ERROR 恢复 20 条：一个 tool_call 改坏 → ERROR → 插恢复句 → 继续推理（半自动，样例抽查）
C 语言一致：非 ASCII >3% 过滤
混合：100 正常 + 15A + 20B + 150 纯 CoT（v2 RFT 原样转换，system 带工具说明但示范无调用）
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_executor import safe_calc
from rollout_multiturn import TOOL_SYSTEM_PROMPT, BOXED_RE, TOOL_CALL_RE
from cold_start_data import latex_to_plain

V1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_coldstart.jsonl")
RFT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gsm8k_rft.jsonl")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_coldstart_v2.jsonl")


def is_pure_english(text, threshold=0.03):
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / max(1, len(text)) <= threshold


def make_error_recovery(content: str) -> str | None:
    """B 类构造：把第一个成功的 tool_call 对改成 失败→ERROR→恢复句→原推理继续。

    返回改写文本；无 tool_call 对返回 None。
    """
    m = re.search(
        r"<tool_call>calc\(([^)]+)\)</tool_call>\s*\n?\s*<tool_result>([^<]+)</tool_result>",
        content,
    )
    if not m:
        return None
    expr, result = m.group(1), m.group(2)
    # 坏表达式：追加 "/0"（除零，tool_executor 真实会抛 ZeroDivisionError）
    bad_expr = expr + "/0"
    try:
        safe_calc(bad_expr)
        return None          # 没抛异常说明这个表达式坏不了，换条数据
    except Exception as e:
        err_msg = str(e)
    recovery = (
        "\nThe calculator failed on that expression. "
        "Let me work through it directly.\n"
    )
    # 原文：tool_call 对替换为 坏调用+ERROR+恢复句；后续推理保持（心算部分本来就是文本）
    broken_pair = (
        f"<tool_call>calc({expr})</tool_call>\n"
        f"<tool_result>{result}</tool_result>"
    )
    new_pair = (
        f"<tool_call>calc({bad_expr})</tool_call>\n"
        f"User: <tool_result>ERROR: {err_msg}</tool_result>\n"
        f"Assistant: {recovery.strip()}"
    )
    return content.replace(broken_pair, new_pair, 1)


def main():
    v1 = [json.loads(l) for l in open(V1_PATH) if l.strip()]
    print(f"v1 数据: {len(v1)} 条")

    # ---- C 语言过滤（全池）----
    clean = [r for r in v1 if is_pure_english(r["completion"][0]["content"])]
    print(f"语言过滤后: {len(clean)} 条（剔除 {len(v1)-len(clean)} 条混杂）")

    # ---- A 完成即停 15 条：completion 截断到最后一个 boxed 闭合 ----
    a_rows = []
    for r in clean:
        content = r["completion"][0]["content"]
        ms = list(BOXED_RE.finditer(content))
        if not ms:
            continue
        trimmed = content[: ms[-1].end()]
        if trimmed == content:
            continue
        a_rows.append({
            "prompt": r["prompt"],
            "completion": [{"role": "assistant", "content": trimmed}],
            "source": "boundary_A_stop",
        })
        if len(a_rows) >= 15:
            break
    print(f"A 完成即停: {len(a_rows)} 条")

    # ---- B ERROR 恢复 20 条 ----
    b_rows = []
    for r in clean:
        if any(r is a for a in [x for x in clean if x in a_rows]):
            continue
        content = r["completion"][0]["content"]
        new_content = make_error_recovery(content)
        if new_content is None:
            continue
        b_rows.append({
            "prompt": r["prompt"],
            "completion": [{"role": "assistant", "content": new_content}],
            "source": "boundary_B_recovery",
        })
        if len(b_rows) >= 20:
            break
    print(f"B ERROR 恢复: {len(b_rows)} 条")

    # ---- 100 正常（排除已被 A/B 用的）----
    used_ids = {id(r) for r in a_rows} | {id(r) for r in b_rows}
    normal = [r for r in clean if id(r) not in used_ids][:100]

    # ---- 150 纯 CoT（v2 RFT 转换，无工具调用，system 带工具说明=可选）----
    cot_rows = []
    with open(RFT_PATH) as f:
        for line in f:
            if len(cot_rows) >= 150:
                break
            ex = json.loads(line)
            comp = (ex.get("completion") or [{}])[0].get("content", "")
            if not comp:
                continue
            plain = latex_to_plain(comp)
            if plain is None or not is_pure_english(plain):
                continue
            cot_rows.append({
                "prompt": ex["prompt"] if isinstance(ex["prompt"], list) else [
                    {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                    {"role": "user", "content": ex["prompt"]},
                ],
                "completion": [{"role": "assistant", "content": plain}],
                "source": "rft_pure_cot",
            })
    print(f"纯 CoT: {len(cot_rows)} 条")

    # ---- 混合输出 ----
    all_rows = normal + a_rows + b_rows + cot_rows
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"cold_start_v2: {len(all_rows)} 条 → {OUT_PATH}")
    # 样例检查：一条 B 类
    if b_rows:
        print("=== B 类样例（ERROR 恢复）===")
        print(b_rows[0]["completion"][0]["content"][:600])


if __name__ == "__main__":
    main()
