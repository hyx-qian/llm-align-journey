# -*- coding: utf-8 -*-
"""第一组实验核心修正：tool_result 移出训练目标（多段样本格式）。

问题：v1-v3 冷启动数据的 completion 里包含 <tool_result>（环境响应）
  → 模型在被训练"自己生成工具结果" → 幻觉 tool_result 的训练来源。

修正：把每条示范按 tool_call 切成多段样本：
  样本 k：prompt = [system, user, assistant段1, user(tool_result1), assistant段2, user(tool_result2)...]
          completion = [assistant: 当前段（推理 + 可选<tool_call>结尾）]
  → 每段 assistant 只含"推理 + 可选 tool_call"，tool_result 永远在 user/tool 轮（环境侧）。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_rollout import TOOL_SYSTEM_PROMPT

V3_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_coldstart_v3.jsonl")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_coldstart_v4_multiturn.jsonl")

TC_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def split_into_segments(completion_text: str):
    """把单段 completion（含 tool_call + tool_result 混排）切成分段结构。

    段的定义：assistant 的连续输出块（以 <tool_call> 结尾或以最终答案结尾）。
    tool_result 视为环境轮（不进任何 assistant 段）。
    Returns: list[dict] [{"assistant": str}, {"env_result": str}, ...] 或 None
    """
    segments = []
    pos = 0
    for m in TC_RE.finditer(completion_text):
        before = completion_text[pos:m.start()]
        # tool_call 之前的 assistant 文字（含工具调用本身——调用是 assistant 行为）
        seg_assistant = before + m.group(0)
        # tool_result 紧随其后
        rest = completion_text[m.end():]
        tr_m = re.match(r"\s*<tool_result>[^<]*</tool_result>", rest)
        if tr_m:
            segments.append({"assistant": seg_assistant.strip()})
            segments.append({"env_result": tr_m.group(0).strip()})
            pos = m.end() + tr_m.end()
        else:
            # tool_call 后没有紧跟 tool_result（格式异常）→ 跳过该条
            return None
    tail = completion_text[pos:]
    if tail.strip():
        segments.append({"assistant": tail.strip()})
    if not any("assistant" in s for s in segments):
        return None
    # 最后一段必须以 \boxed 结尾（完整任务）
    if "\\boxed" not in segments[-1].get("assistant", ""):
        return None
    return segments


def build_multiturn_samples(row):
    """把一条示范的多段结构转成多条训练样本（每段一个样本，prompt 累积历史）。"""
    segs = split_into_segments(row["completion"][0]["content"])
    if segs is None:
        return []
    system_msg = row["prompt"][0]
    user_msg = next((m for m in row["prompt"] if m["role"] == "user"), None)
    samples = []
    history = [system_msg, user_msg]
    for idx, seg in enumerate(segs):
        if "env_result" in seg:
            history.append({"role": "user", "content": seg["env_result"]})
            continue
        # assistant 段 → 训练样本（prompt=累积历史, completion=本段）
        samples.append({
            "prompt": list(history),
            "completion": [{"role": "assistant", "content": seg["assistant"]}],
            "segment": idx + 1,
            "source": row.get("source", "coldstart") + "_multiturn",
        })
        history.append({"role": "assistant", "content": seg["assistant"]})
    return samples


def main():
    rows = [json.loads(l) for l in open(V3_PATH) if l.strip()]
    print(f"输入: {len(rows)} 条示范")
    all_samples = []
    n_skip = 0
    for r in rows:
        samples = build_multiturn_samples(r)
        if not samples:
            n_skip += 1
            continue
        all_samples.extend(samples)
    # 上采样保持 8:1 思路：多段样本自然扩充（每条示范产生 2-4 个样本）
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    # 统计：completion 里不应再有任何 tool_result
    leak = sum(1 for s in all_samples if "<tool_result>" in s["completion"][0]["content"])
    print(f"输出: {len(all_samples)} 条多段样本（跳过 {n_skip} 条不完整示范）")
    print(f"completion 含 tool_result 泄漏: {leak} 条（必须为 0）")
    print(f"平均每示范段数: {len(all_samples)/max(1, len(rows)-n_skip):.1f}")
    print(f"→ {OUT_PATH}")
    if all_samples:
        print("=== 样例（第 1 条样本的 completion）===")
        print(all_samples[0]["completion"][0]["content"][:400])


if __name__ == "__main__":
    main()
