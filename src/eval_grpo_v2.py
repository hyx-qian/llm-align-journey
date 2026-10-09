# -*- coding: utf-8 -*-
"""RFT-SFT test 集评测：vLLM 原生 LoRA 路径（绕开 merge bug）
放服务器 src/ 下，在 align-eval 环境跑：
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python eval_rft.py
"""
import json

from datasets import load_dataset

import gsm8k_eval

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "exp/grpo_v2_rft/checkpoint-1000"
DATA = "data/gsm8k_sft_test.jsonl"
N = 1319


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ds = load_dataset("json", data_files=DATA, split="train").select(range(N))
    convs = [[ex["prompt"][0]] for ex in ds]

    llm = LLM(model=BASE,                 # 基座从 hub 缓存读（tokenizer 也是原版，避开 5.16 格式坑）
              enable_lora=True,           # 原生 LoRA：adapter 运行时挂载，不 merge
              max_lora_rank=16,           # 必须 >= adapter 的 r（r8 → 16 稳）
              dtype="bfloat16",
              gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0, max_tokens=512)   # 贪心，与 base/r8/r16 完全一致（可比性）
    outputs = llm.chat(convs, sp,
                       lora_request=LoRARequest("grpo", 1, ADAPTER))

    results, correct, wrong_shown = [], 0, 0
    for ex, o in zip(ds, outputs):
        pred_text = o.outputs[0].text
        gold_text = ex["completion"][0]["content"]
        ok = gsm8k_eval.score(pred_text, gold_text)
        correct += int(ok)
        results.append({"question": ex["prompt"][0]["content"][:80],
                        "pred": pred_text[-200:], "gold": gold_text[-80:],
                        "correct": ok})
        if not ok and wrong_shown < 5:   # 前 5 条错题打出来（failure case 素材）
            wrong_shown += 1
            print("---- WRONG ----\nPRED:", pred_text[-300:], "\nGOLD:", gold_text[-120:])

    acc = correct / len(results)
    with open("exp/eval_results_grpo.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"GRPO_ACC = {acc:.4f}  ({correct}/{len(results)})")


if __name__ == "__main__":
    main()
