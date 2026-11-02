# -*- coding: utf-8 -*-
"""SFT 冷启动验证：smoke 10 题，判定工具调用率（baseline 0% → 目标 >80%）。

用 vLLM 原生 LoRA 推理（enable_lora）——规避 merge_and_unload 历史前科。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 离线用本地缓存（服务器到 HF 不通）
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout_multiturn import rollout_multiturn
from reward_agent import combined_reward

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

BASE = "/home/huyx/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ADAPTER = "/home/huyx/study/projects/llm-align-journey/src/agent_rl/exp/agent_sft_coldstart_v2/checkpoint-27"
AGENT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_agent_train_4k.jsonl")

N_TEST = 50


class VLLMLoRAClient:
    """带 LoRA adapter 的 vLLM client（接口契约与 FakeLLM 一致：可调用，返回 str）。"""

    def __init__(self, base_model: str, adapter_path: str):
        self.llm = LLM(model=base_model, dtype="bfloat16",
                       gpu_memory_utilization=0.35, enable_lora=True, max_lora_rank=16)
        self.sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
        self.lora_req = LoRARequest("coldstart", 1, adapter_path)

    def __call__(self, messages: list, sampling_params=None) -> str:
        sp = sampling_params or self.sp
        outputs = self.llm.chat([messages], sp, lora_request=self.lora_req)
        return outputs[0].outputs[0].text


def main():
    rows = [json.loads(l) for l in open(AGENT_DATA) if l.strip()][:N_TEST]
    print(f"verify_coldstart: {len(rows)} 题")

    client = VLLMLoRAClient(BASE, ADAPTER)

    n_tool = n_correct = 0
    results = []
    for i, ex in enumerate(rows):
        user_msg = next((m["content"] for m in ex["prompt"] if m["role"] == "user"), "")
        gold = ex.get("gold_answer", "")
        try:
            tr = rollout_multiturn(client, user_msg, sampling_params=None)
            reward = combined_reward(tr, gold)
        except Exception as e:
            print(f"[{i}] ERROR: {type(e).__name__} {e}")
            continue
        used = tr.get("num_tool_calls", 0) > 0
        correct = reward >= 1.0
        n_tool += int(used)
        n_correct += int(correct)
        results.append({"idx": i, "num_tool_calls": tr.get("num_tool_calls"),
                        "turns_used": tr.get("turns_used"), "final_answer": tr.get("final_answer"),
                        "reward": reward, "used_tool": used, "correct": correct,
                        "trajectory_tail": [c[-150:] for _, c in tr.get("trajectory", [])[-3:]]})
        print(f"[{i}] tool={tr.get('num_tool_calls')} turns={tr.get('turns_used')} "
              f"ans={tr.get('final_answer')} reward={reward:.2f}")

    n = len(results)
    print("=" * 50)
    print(f"工具调用率: {n_tool}/{n} = {n_tool/max(1,n)*100:.0f}%   （baseline 0%，目标 >80%）")
    print(f"判对率:     {n_correct}/{n} = {n_correct/max(1,n)*100:.0f}%")
    print("判定:", "✅ SFT 冷启动成功（Agent RL 阶段 1 闭环）" if n_tool / max(1, n) >= 0.8
          else "⚠️ 冷启动部分生效（>0 但 <80%，可加 epoch 或数据）"
          if n_tool > 0 else "❌ 冷启动失败（仍 0%，检查 adapter 路径/格式）")

    with open("results/verify_coldstart.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
