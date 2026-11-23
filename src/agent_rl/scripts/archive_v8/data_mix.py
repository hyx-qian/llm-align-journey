# -*- coding: utf-8 -*-
"""冷启动数据配比 v3：工具示范上采样 ×3（135→405）+ 纯 CoT 砍到 50。

背景：v2 配比（135 工具示范 47% + 150 纯 CoT 53%）训出的模型学成"工具可选"
→ 工具调用率 0%（v2 验证）——目标"必须用工具"的行为需要 8:1 强配比。
产出：gsm8k_coldstart_v3.jsonl（455 条 = 405 工具示范 + 50 纯 CoT）。
可复现：固定 seed=42。
"""
import json
import os
import random
from collections import Counter

V2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_coldstart_v2.jsonl")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_coldstart_v3.jsonl")
PURE_COT_KEEP = 50        # 防遗忘的纯 CoT 保留量
SEED = 42


def main():
    recs = [json.loads(l) for l in open(V2_PATH) if l.strip()]
    tool_demo = [r for r in recs if "<tool_call>" in r["completion"][0]["content"]]
    pure_cot = [r for r in recs if "<tool_call>" not in r["completion"][0]["content"]]
    print(f"v2 输入: 工具示范 {len(tool_demo)} / 纯CoT {len(pure_cot)}")

    random.seed(SEED)
    upsampled = []
    for r in tool_demo:
        upsampled.extend([r] * 3)          # ×3 上采样
    random.shuffle(upsampled)              # 打散重复
    v3 = upsampled + pure_cot[:PURE_COT_KEEP]
    random.shuffle(v3)

    with open(OUT_PATH, "w") as f:
        for r in v3:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    has_tool = sum(1 for r in v3 if "<tool_call>" in r["completion"][0]["content"])
    print(f"v3 输出: {len(v3)} 条 = 工具示范 {has_tool} ({has_tool/len(v3)*100:.0f}%) "
          f"+ 纯CoT {len(v3)-has_tool}  → {OUT_PATH}")
    print(f"来源: {dict(Counter(r.get('source','?') for r in v3))}")


if __name__ == "__main__":
    main()
