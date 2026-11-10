# -*- coding: utf-8 -*-
"""Agent RL 阶段 1 数据改造：GSM8K prompt 加工具说明（v2 数据 → agent_v1 数据）。

从 src/data/gsm8k_grpo_train_4k.jsonl 读取 v2 的训练数据，
为每条 prompt 加上工具使用说明（system prompt），
产出 src/agent_rl/data/gsm8k_agent_train_4k.jsonl。
"""
import json
import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_rollout import TOOL_SYSTEM_PROMPT

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gsm8k_grpo_train_4k.jsonl")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_agent_train_4k.jsonl")




def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    n_in, n_out = 0, 0
    with open(SRC) as f, open(DST, "w") as out:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            n_in += 1
            # v2 数据结构：prompt[0] 是 user 轮，completion[0] 是 gold assistant 轮
            # agent_v1 数据结构：prompt 加 system 轮（工具说明），user 轮不变
            user_msg = None
            if "prompt" in ex:
                for m in ex["prompt"]:
                    if m.get("role") == "user":
                        user_msg = m["content"]
                        break
            if user_msg is None:
                continue
            new_ex = {
                "prompt": [
                    {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                # completion 保留 v2 的 gold（仅供 reference，训练时 GRPO 在线采样）
                "gold_answer": ex.get("gold_answer", ""),
                "source": "gsm8k_agent_v1",
            }
            out.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"data_agent: {n_in} -> {n_out} 条 -> {DST}")


if __name__ == "__main__":
    main()
