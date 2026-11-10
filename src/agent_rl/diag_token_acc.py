# -*- coding: utf-8 -*-
"""冷启动深度诊断：
  A. 训练分布内生成：用 coldstart 数据第 1 条的 prompt（与训练完全相同的渲染）贪心生成
     —— 看是否输出 <tool_call>（训练分布内的行为）
  B. 分位置 token accuracy：teacher-force 示范轨迹，
     统计 <tool_call> 标记处 vs 其他位置的 next-token 准确率
     —— 验证"90.8% 被稀释"的假设
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import json
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/home/huyx/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ADAPTER = "/home/huyx/study/projects/llm-align-journey/src/agent_rl/exp/agent_sft_coldstart_v2/checkpoint-27"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_coldstart.jsonl")

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

rows = [json.loads(l) for l in open(DATA) if l.strip()]
row = rows[0]   # 训练时的第一条
prompt_msgs = row["prompt"]
completion = row["completion"][0]["content"]

# ========== A. 训练分布内生成（贪心） ==========
print("=" * 20, "A. 训练分布内贪心生成", "=" * 20)
text = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=700, do_sample=False, pad_token_id=tok.eos_token_id)
gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
print(gen[:800])
print("含 <tool_call>:", "<tool_call>" in gen, "| 含 \\boxed:", "boxed" in gen)

# ========== B. 分位置 token accuracy ==========
print("=" * 20, "B. 分位置 token accuracy", "=" * 20)
# teacher-force：prompt 渲染 + completion 拼接，算 completion 每个位置的 next-token 预测是否正确
full_text = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False) + completion
ids = tok(full_text, return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    logits = model(ids).logits
pred = logits[0, :-1].argmax(dim=-1)
true = ids[0, 1:]
correct = (pred == true).tolist()

# 标注 <tool_call> 标记覆盖的位置（在 token 序列里找 tool_call 的字符范围）
tc_spans = []
for m in __import__("re").finditer(r"<tool_call>.*?</tool_call>", completion, __import__("re").DOTALL):
    # 找到 completion 在 full_text 中的偏移（简单法：full_text 从 prompt 渲染后开始）
    start_char = full_text.index(m.group(0), len(tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False))) if m.group(0) in full_text else -1
    if start_char >= 0:
        tc_spans.append((start_char, start_char + len(m.group(0))))

# 字符范围 → token 范围（用 offset mapping 重算更准，这里用近似：重分词 completion 段）
# 简化：直接把 completion 重新分词，对齐 full_text 的尾部
comp_ids = tok(completion, add_special_tokens=False).input_ids
n_comp = len(comp_ids)
# completion 对应 full_text 的最后 n_comp 个 token（近似，可能有边界误差）
comp_correct = correct[-n_comp:] if n_comp <= len(correct) else correct
# tool_call 标记的 token 范围（在 completion 子序列中找 '<tool_call>' 的 token）
tc_tok_positions = []
for m in __import__("re").finditer(r"<tool_call>", completion):
    prefix = completion[:m.start()]
    n_before = len(tok(prefix, add_special_tokens=False).input_ids)
    tc_len = len(tok("<tool_call>", add_special_tokens=False).input_ids)
    tc_tok_positions.extend(range(n_before, n_before + tc_len))

tc_correct = [comp_correct[i] for i in tc_tok_positions if i < len(comp_correct)]
other_correct = [c for i, c in enumerate(comp_correct) if i not in set(tc_tok_positions)]
print(f"completion token 数（近似）: {n_comp}")
print(f"<tool_call> 标记处: {len(tc_correct)} tokens, 准确率 {sum(tc_correct)/max(1,len(tc_correct))*100:.1f}%")
print(f"其他位置:           {len(other_correct)} tokens, 准确率 {sum(other_correct)/max(1,len(other_correct))*100:.1f}%")
print("→ 若标记处显著低于其他位置 = '格式学习不足被平均稀释'假设成立")
