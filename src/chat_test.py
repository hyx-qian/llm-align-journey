import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 双保险，不依赖shell变量

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("开始下载/加载模型（约1GB，走国内镜像很快）……")
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")

messages = [
    {"role": "system", "content": "你是一个简洁、专业的AI助手。"},
    {"role": "user",   "content": "用一句话解释什么是强化学习。"},
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128)

reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n===== 模型回复 =====")
print(reply)
