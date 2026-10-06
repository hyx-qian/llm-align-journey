# -*- coding: utf-8 -*-
"""DPO 偏好对数据管线（升级版：含原始样本全量落盘）

设计原则（9.29 教训）：**采样贵、筛选便宜——一次采集，全量落盘，多次消费。**
RFT 那次采样把 ~45% 的错误输出直接丢了，导致 DPO 要重新采一遍——
这次把原始输出全部存盘，RFT / DPO / W6 错误分析 / GRPO 参考都从同一份原始数据加工。

四步：
  1. 采样：base 对 train 前 5000 题 × k=4（temperature=1.0 多样性）
  2. 落盘：data/gsm8k_samples_raw.jsonl —— 每题一行：题目 + 全部原始输出 + 判分标签
  3. 配对：chosen = 正确且含 \boxed；rejected = 含 \boxed 但答案错；每题最多 1 对
  4. 产出：data/gsm8k_dpo.jsonl（TRL DPOTrainer 的 conversational 偏好格式）

用法：align-eval 环境，tmux 里跑，约 30 分钟
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python dpo_data.py
"""
import json

from datasets import load_dataset
from vllm import LLM, SamplingParams

import gsm8k_eval

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
K = 8
N_QUESTIONS = 7473
RAW_PATH = "data/gsm8k_samples_raw_k8.jsonl"
OUT_PATH = "data/gsm8k_dpo_k8.jsonl"


def main():
    ds = load_dataset("json", data_files="data/gsm8k_sft_train.jsonl",
                      split="train").select(range(N_QUESTIONS))
    convs = [[ex["prompt"][0]] for ex in ds]

    llm = LLM(model=BASE, dtype="bfloat16", gpu_memory_utilization=0.85)
    sp = SamplingParams(n=K, temperature=1.0, top_p=0.95, max_tokens=512)   # 多样性采样
    outputs = llm.chat(convs, sp)

    # ---------- 第 2 步：判分 + 全量落盘（TODO-1，约 8 行） ----------
    # 逐条 gsm8k_eval.score 判分。
    # 每题写一行到 RAW_PATH：
    #   {"question": 原题, "outputs": [归一化后的 k 条全文], "scores": [true/false...]}
    # raise NotImplementedError("TODO-1: 判分并全量落盘")
    with open(RAW_PATH, 'w') as f:
        for idx, (gold, o) in enumerate(zip(ds, outputs)):
            row = {
                'question': gold['prompt'],
                'outputs': [],
                'scores': [],
                'native_boxed': []
            }
            gold_text = gold['completion'][0]['content']
            for pred in o.outputs:
                pred_text = pred.text
                row['outputs'].append(pred_text)
                row['native_boxed'].append('\\boxed' in pred_text)
                row['scores'].append(gsm8k_eval.score(pred_text, gold_text))  # extract 自带兜底，仅作统计
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
                
                
            

    # ---------- 第 3 步：配对（TODO-2，约 15 行） ----------
    # 重新读 RAW_PATH（就当它是别人的数据），对每题：
    #   rights  = [o for o, s in zip(outputs, scores) if s]        # 正确
    #   wrongs  = [o for o, s in zip(outputs, scores) if not s]    # 错误
    #   chosen   = rights 里第一条（都含 boxed，归一化保证过）
    #   rejected = wrongs 里第一条（格式崩坏的已排除——归一化保证过）
    #   两边都非空才写一对：{"prompt": [...], "chosen": [...], "rejected": [...]}
    #   （conversational 格式：chosen/rejected 都是 [{"role":"assistant","content":...}]）
    # 最后打印：配对率 = 配成对题数 / N_QUESTIONS（预期 ~85%）、总对数
    # raise NotImplementedError("TODO-2: 配对并写出 jsonl")
    
    with open(RAW_PATH) as f:
        rows = [json.loads(line) for line in f]

    pair = 0
    with open(OUT_PATH, "w") as f:
        for row in rows:
            rights = [o for o, s, nv in zip(row["outputs"], row["scores"], row["native_boxed"]) if s and nv]
            wrongs = [o for o, s, nv in zip(row["outputs"], row["scores"], row["native_boxed"]) if not s and nv]
            if not rights or not wrongs:      # 缺任一边直接跳过，杜绝泄漏
                continue
            pair += 1
            f.write(json.dumps({
                "prompt": row["question"],
                "chosen":   [{"role": "assistant", "content": rights[0]}],
                "rejected": [{"role": "assistant", "content": wrongs[0]}],
            }, ensure_ascii=False) + "\n")
        print(f'配对率 = {pair / len(rows) * 100:.1f}%, 总对数 = {pair}')


if __name__ == "__main__":
    main()
