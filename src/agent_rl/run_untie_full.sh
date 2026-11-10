#!/bin/bash
# Agent RL untie 全链：untie 配置 → 训练 → merge → 双加载器验证
# 全程 tee 日志到 results/，可复现
set -e
cd ~/study/projects/llm-align-journey/src/agent_rl
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
LOG="results/untie_chain_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

BASE_LOCAL="/home/huyx/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PY=/home/huyx/miniforge3/envs/align/bin/python

echo "=== Step 1: sft_agent.py untie 修改 ==="
$PY - << 'PYEOF'
import re
src = open('sft_agent.py').read()
# 1. model 改本地 snapshot（网络断保险 + 离线一致）
src = re.sub(r'model="[^"]+"',
             'model="/home/huyx/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"',
             src, count=1)
# 2. SFTConfig 加 untie（embed/lm_head 独立初始化）
if 'tie_word_embeddings' not in src:
    src = src.replace('    max_length=2048',
                      '    max_length=2048,\n    model_init_kwargs={"tie_word_embeddings": False},  # untie: embed/lm_head 独立（merge bug 修复）')
# 3. output_dir → untie 版本
src = src.replace('exp/agent_sft_coldstart_v4', 'exp/agent_sft_untie')
open('sft_agent.py', 'w').write(src)
import ast; ast.parse(src)
print('untie config OK (model=local snapshot, tie=False, out=agent_sft_untie)')
PYEOF

echo "=== Step 2: 训练 v5（untie + modules_to_save，~2 分钟）==="
$PY sft_agent.py 2>&1 | grep -vE 'it/s\]|it\] ' | tail -4

echo "=== Step 3: merge v5 ==="
FINAL=$(ls -d exp/agent_sft_untie/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "checkpoint: $FINAL"
$PY - << PYEOF
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
BASE = "$BASE_LOCAL"
ADAPTER = "$FINAL"
OUT = "exp/agent_sft_untie_merged"
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
merged.save_pretrained(OUT)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print("MERGED ->", OUT)
PYEOF

echo "=== Step 4: 双加载器验证 A（transformers 加载 merged，10 题）==="
$PY - << PYEOF
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
MERGED = "exp/agent_sft_untie_merged"
tok = AutoTokenizer.from_pretrained(MERGED)
model = AutoModelForCausalLM.from_pretrained(MERGED, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
rows = [json.loads(l) for l in open("data/gsm8k_agent_train_4k.jsonl") if l.strip()][:10]
n_tool = n_correct = 0
for i, ex in enumerate(rows):
    user_msg = next((m["content"] for m in ex["prompt"] if m["role"] == "user"), "")
    gold = ex.get("gold_answer", "")
    messages = [{"role": "system", "content": "You are a math assistant with access to a calculator tool."},
                {"role": "user", "content": user_msg}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    has_call = "<tool_call>" in gen
    nums = re.findall(r"-?\d+(?:\.\d+)?", gen.replace(",", ""))
    gnums = re.findall(r"-?\d+(?:\.\d+)?", gold.replace(",", ""))
    correct = bool(nums and gnums and abs(float(nums[-1]) - float(gnums[-1])) < 1e-6)
    n_tool += int(has_call); n_correct += int(correct)
    print(f"[{i}] tool_call={has_call} ans={nums[-1] if nums else None} correct={correct}")
print(f"[A] transformers+untie_merged: 工具调用 {n_tool}/10, 判对 {n_correct}/10")
print("[A] → 若 ~5/10 = untie 修复闭环（merge 归因完成）；~0 = untie 不够，改 PEFT 推理做迭代")
PYEOF

echo "=== Step 5: 双加载器验证 B（vLLM 加载 merged，同 10 题）==="
$PY - << PYEOF
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import json, re
from vllm import LLM, SamplingParams
MERGED = "exp/agent_sft_untie_merged"
llm = LLM(model=MERGED, dtype="bfloat16", gpu_memory_utilization=0.35)
sp = SamplingParams(temperature=0.0, max_tokens=512)
tok = llm.get_tokenizer()
rows = [json.loads(l) for l in open("data/gsm8k_agent_train_4k.jsonl") if l.strip()][:10]
n_tool = n_correct = 0
for i, ex in enumerate(rows):
    user_msg = next((m["content"] for m in ex["prompt"] if m["role"] == "user"), "")
    gold = ex.get("gold_answer", "")
    messages = [{"role": "system", "content": "You are a math assistant with access to a calculator tool."},
                {"role": "user", "content": user_msg}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    out = llm.generate([text], sp)
    gen = out[0].outputs[0].text
    has_call = "<tool_call>" in gen
    nums = re.findall(r"-?\d+(?:\.\d+)?", gen.replace(",", ""))
    gnums = re.findall(r"-?\d+(?:\.\d+)?", gold.replace(",", ""))
    correct = bool(nums and gnums and abs(float(nums[-1]) - float(gnums[-1])) < 1e-6)
    n_tool += int(has_call); n_correct += int(correct)
    print(f"[{i}] tool_call={has_call} ans={nums[-1] if nums else None} correct={correct}")
print(f"[B] vLLM+untie_merged: 工具调用 {n_tool}/10, 判对 {n_correct}/10")
print("→ A/B 一致 = merge 产物干净，untie 修复在两个加载器下都成立（W5 前科归因完成）")
PYEOF

echo "=== CHAIN COMPLETE ==="
