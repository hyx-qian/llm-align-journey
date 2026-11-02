# -*- coding: utf-8 -*-
"""SFT 冷启动验证（PEFT 推理版）：transformers+PeftModel 加载 adapter——
训练同款加载方式，排除 vLLM LoRA 推理路径的因素。10 题快速判定。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import json
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rollout_multiturn import TOOL_SYSTEM_PROMPT

BASE = "/home/huyx/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ADAPTER = "/home/huyx/study/projects/llm-align-journey/src/agent_rl/exp/agent_sft_coldstart_v2/checkpoint-27"
AGENT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_agent_train_4k.jsonl")

N_TEST = 10


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print("PEFT model loaded")

    rows = [json.loads(l) for l in open(AGENT_DATA) if l.strip()][:N_TEST]

    n_tool = n_correct = n_done = 0
    for i, ex in enumerate(rows):
        user_msg = next((m["content"] for m in ex["prompt"] if m["role"] == "user"), "")
        gold = ex.get("gold_answer", "")
        # 第一轮生成（贪心，最大确定性）
        messages = [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

        has_call = "<tool_call>" in gen
        has_boxed = "\\boxed{" in gen or "\\boxed" in gen
        # 简易判对：最后一个数字与 gold 的 ####/boxed 数字比对
        import re
        nums = re.findall(r"-?\d+(?:\.\d+)?", gen.replace(",", ""))
        gnums = re.findall(r"-?\d+(?:\.\d+)?", gold.replace(",", ""))
        correct = bool(nums and gnums and abs(float(nums[-1]) - float(gnums[-1])) < 1e-6)
        n_tool += int(has_call)
        n_correct += int(correct)
        n_done += 1
        print(f"[{i}] tool_call={'Y' if has_call else 'N'} boxed={'Y' if has_boxed else 'N'} "
              f"ans={nums[-1] if nums else None} gold={gnums[-1] if gnums else None} correct={correct}")

    print("=" * 50)
    print(f"工具调用率: {n_tool}/{n_done} = {n_tool/max(1,n_done)*100:.0f}%   （base baseline 0%）")
    print(f"判对率:     {n_correct}/{n_done} = {n_correct/max(1,n_done)*100:.0f}%")
    print("判定:", "✅ adapter 生效（agent RL 实证到手）" if n_tool / max(1, n_done) >= 0.5
          else "❌ adapter 未生效（PEFT 路径也 0% = SFT 训练本身的问题）")


if __name__ == "__main__":
    main()
