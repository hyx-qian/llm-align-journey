# -*- coding: utf-8 -*-
"""RFT 拒绝采样数据管线：base 模型自产高质量 CoT，修复朴素 SFT 的能力侵蚀

四步：
  1. 采样：base 对 train 7473 题 × k=8（temperature=1.0 保多样性）
  2. 判分：复用 gsm8k_eval.score
  3. 筛选：每题保留 ≤2 条正确且去重的 CoT（防简单题刷屏）
  4. 产出：data/gsm8k_rft.jsonl（格式与 sft 完全一致）+ coverage 统计

用法：python rft_data.py          # 全量采样约 1-1.5h，tmux 里跑
"""
import json
import re
from collections import defaultdict

from datasets import load_dataset
from vllm import LLM, SamplingParams

import gsm8k_eval

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
K = 8                     # 每题采样次数
MAX_KEEP = 2              # 每题最多保留条数
OUT_PATH = "data/gsm8k_rft.jsonl"


def build_conversations(ds):
    return [[ex["prompt"][0]] for ex in ds]


def keep_correct(ds, outputs):
    """TODO-1: 判分+筛选
    对每道题：
      a) 用 gsm8k_eval.score(gen_text, gold_text) 判对错（gold 从 ex['completion'][0]['content'] 取）
      b) 收集答对样本的 (CoT全文, 抽取的答案)
      c) 去重：CoT 前 50 字符相同视为重复（同一解法采样多次）
      d) 每题最多保留 MAX_KEEP 条
    返回: kept = {题号: [completion_content, ...]}，并打印 coverage 统计
    coverage = 至少有一条正确解的题数 / 总题数（RFT 的关键健康指标）
    """
    # raise NotImplementedError("TODO-1")
    kept = {}
    coverage = 0
    for idx, (ex, o) in enumerate(zip(ds, outputs)):
        now_correct = 0
        CoTs = set()
        kept[idx] = []
        for pred in o.outputs:
            if now_correct >= MAX_KEEP:
                break
            pred_text = pred.text
            if '\\boxed' not in pred_text:
                ans = gsm8k_eval.extract_answer(pred_text) or ''
                pred_text = pred_text + f'\n\\boxed{{{ans}}}'
            gold_text = ex["completion"][0]["content"]
            if gsm8k_eval.score(pred_text, gold_text):
                CoT = pred_text[:50]
                if CoT not in CoTs:
                    now_correct += 1
                    CoTs.add(CoT)
                    kept[idx].append(pred_text)
        if now_correct > 0:
            coverage += 1
    return kept, coverage


def main():
    ds = load_dataset("json", data_files="data/gsm8k_sft_train.jsonl", split="train")
    convs = build_conversations(ds)

    llm = LLM(model=BASE, dtype="bfloat16", gpu_memory_utilization=0.85)
    # TODO-2: SamplingParams——注意与评测不同：这里要多样性，temperature=1.0, top_p=0.95, max_tokens=512
    sp = SamplingParams(n=K, temperature=1.0, top_p=0.95, max_tokens=512)
    outputs = llm.chat(convs, sp)

    kept, coverage = keep_correct(ds, outputs)
    with open(OUT_PATH, "w") as f:
        for qid in sorted(kept):
            ex = ds[qid]
            for cot in kept[qid]:
                row = {"prompt": ex["prompt"],
                       "completion": [{"role": "assistant", "content": cot}]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"coverage: {coverage}/{len(ds)} = {coverage/len(ds):.2%}")
    print(f"总保留条数: {sum(len(v) for v in kept.values())} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
