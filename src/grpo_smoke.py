# -*- coding: utf-8 -*-
"""GRPO 冒烟骨架：0.5B + LoRA + GSM8K 可验证 reward（方案 A：纯判对错）

发车前三件事：
  1) inspect.signature(GRPOConfig.__init__) 核对参数名（TRL 1.12，标注 ★ 的以查签名为准）
  2) nvidia-smi 侦察：独占卡才发车（本预算 ~13-14GB）
  3) 数据：data/gsm8k_grpo_train.jsonl —— 从 gsm8k_sft_train.jsonl 转一列 gold_answer

用法：align 环境（训练）+ colocate vLLM（rollout），tmux 里跑
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python grpo_smoke.py
"""
import json

from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

import gsm8k_eval

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
N_QUESTIONS = 500          # 冒烟规模：500 题 × G=8 = 4000 条 rollouts


# ---------- 数据：加一列 gold_answer（reward 函数要用） ----------
def build_dataset():
    ds = load_dataset("json", data_files="data/gsm8k_sft_train.jsonl", split="train")
    ds = ds.select(range(N_QUESTIONS))
    rows = []
    for ex in ds:
        gold_text = ex["completion"][0]["content"]      # 官方 CoT + \\boxed{ans}
        rows.append({"prompt": ex["prompt"],            # conversational prompt
                     "gold_answer": gold_text})         # 额外列 → TRL 经 kwargs 传给 reward_fn
    return Dataset.from_list(rows)


# ---------- reward：方案 A（纯判对错 1.0/0.0） ----------
def reward_correct(completions, gold_answer, **kwargs):
    """conversational 格式：completions 每条是 [{'role':'assistant','content':...}]"""
    texts = []
    for c in completions:
        if isinstance(c, list):                      # conversational → 抽 content
            texts.append("".join(m.get("content", "") for m in c))
        else:                                        # standard → 原样
            texts.append(c)
    return [1.0 if gsm8k_eval.score(t, g) else 0.0
            for t, g in zip(texts, gold_answer)]



def main():
    # train_ds = build_dataset()
    train_ds = load_dataset("json", data_files="data/gsm8k_grpo_train.jsonl", split="train")

    cfg = GRPOConfig(
        output_dir="exp/grpo_smoke",
        save_strategy="steps", save_steps=50,
        num_generations=8,                    # ★ G=8（手推的组）
        per_device_train_batch_size=8,        # completions 数：1 prompt × 8 gen
        gradient_accumulation_steps=8,        # 有效 64 completions = 8 prompts/优化步
        num_completions_to_print=2,
        max_completion_length=512,
        learning_rate=1e-5,                   # RL 梯度更野，比 DPO 的 5e-5 再小一半
        beta=0.04,                            # ★ 显式！TRL 默认 0.0 = 无缰绳
        epsilon=0.2,                          # clip 范围
        num_iterations=1,                     # ★ μ=1：冒烟先单步（clip 不生效，最稳）
        use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,     # 四本账第 3 本：独占卡给 8GB
        bf16=True,
        logging_steps=5,
        num_train_epochs=1,
    )

    trainer = GRPOTrainer(
        model=BASE,
        args=cfg,
        train_dataset=train_ds,
        reward_funcs=[reward_correct],        # 可验证 reward：你的 gsm8k_eval 会师
        peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                               task_type="CAUSAL_LM"),
    )
    trainer.train()


if __name__ == "__main__":
    main()
