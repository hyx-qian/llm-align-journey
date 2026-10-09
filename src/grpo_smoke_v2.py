# -*- coding: utf-8 -*-
"""GRPO v2：串联路线——从 RFT-SFT checkpoint 续训 + Format reward 维度 + 训练量翻倍

v1 → v2 改动（对照 RESULTS.md"下一步"）：
  1. 起点：base → RFT-SFT checkpoint（PeftModel 挂 adapter，is_trainable=True）；
     π_ref 自动 deepcopy = RFT-SFT 模型（缰绳拴在 RFT 家底上）
  2. reward：方案 A（纯判对错）→ 方案 B（判对 1.0 + 格式 0.1，reward_weights 加权）
  3. 数据：2000 → 4000 题（gsm8k_grpo_train_4k.jsonl，不存在则自动生成）
  4. epochs：1 → 2（训练量翻倍，盯 entropy 刹车：掉到 0.15 人工停）
  5. 不传 peft_config：model 已是 PeftModel，TRL 直接在其上继续训练

用法：align 环境，独占卡（预算 ~18GB），tmux：
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python grpo_smoke_v2.py
"""
import os

from datasets import load_dataset, Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

import gsm8k_eval

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
RFT_ADAPTER = "exp/sft_full_rft/checkpoint-513"      # RFT-SFT 最终档（3ep 收尾）
N_QUESTIONS = 4000
DATA_PATH = "data/gsm8k_grpo_train_4k.jsonl"


# ---------- 数据：4000 题 + gold_answer 列（落盘，版本化） ----------
def build_dataset():
    if os.path.exists(DATA_PATH):
        return load_dataset("json", data_files=DATA_PATH, split="train")
    ds = load_dataset("json", data_files="data/gsm8k_sft_train.jsonl", split="train")
    ds = ds.select(range(N_QUESTIONS))
    rows = [{"prompt": ex["prompt"],
             "gold_answer": ex["completion"][0]["content"]} for ex in ds]
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return load_dataset("json", data_files=DATA_PATH, split="train")


# ---------- reward：方案 B（判对 1.0 + 格式 0.1，双函数加权） ----------
def _to_text(c):
    return "".join(m.get("content", "") for m in c) if isinstance(c, list) else c

def reward_correct(completions, gold_answer, **kwargs):
    return [1.0 if gsm8k_eval.score(_to_text(c), g) else 0.0
            for c, g in zip(completions, gold_answer)]

def reward_format(completions, **kwargs):
    return [1.0 if "\\boxed" in _to_text(c) else 0.0 for c in completions]


def main():
    import torch
    train_ds = build_dataset()

    # ---- 起点串联：基座 + RFT adapter（可训练）----
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, RFT_ADAPTER, is_trainable=True)

    cfg = GRPOConfig(
        output_dir="exp/grpo_v2_rft",
        save_strategy="steps", save_steps=100,
        num_generations=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,        # 有效 64 completions = 8 prompts/步
        num_completions_to_print=2,
        max_completion_length=512,
        learning_rate=1e-5,
        beta=0.04,                            # ★ 缰绳拴在 RFT-SFT 模型上（自动 deepcopy）
        epsilon=0.2,
        num_iterations=1,
        use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        reward_weights=[1.0, 0.1],            # correct 主信号 + format 轻量塑形
        bf16=True,
        logging_steps=5,
        num_train_epochs=2,                   # ★ 训练量翻倍，entropy<0.15 人工停
    )

    trainer = GRPOTrainer(
        model=model,                          # ← 已挂 RFT adapter 的 PeftModel
        args=cfg,
        train_dataset=train_ds,
        reward_funcs=[reward_correct, reward_format],
        # ★ 不传 peft_config：model 已 peft 化，TRL 直接续训这组 adapter
    )
    trainer.train()


if __name__ == "__main__":
    import json
    main()
