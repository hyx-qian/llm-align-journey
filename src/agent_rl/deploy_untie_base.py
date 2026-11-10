# -*- coding: utf-8 -*-
"""方法 1：创建 untied base（lm_head = embed_tokens 的【拷贝】，不是随机初始化）。

背景：Qwen2.5-1.5B 是 tie_word_embeddings=True（embed/lm_head 共享权重）。
  - 直接 tie=False 加载 → lm_head 随机初始化 → 训练从垃圾起点崩溃（11/5 失败原因）
  - 正确做法：tie=False + **lm_head.weight = embed_tokens.weight.clone()**（等价初始化）
产出：qwen1.5b_untied/（完整模型目录，可被 sft_agent.py / vLLM 直接加载）
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/huyx/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OUT = "/home/huyx/study/projects/llm-align-journey/src/agent_rl/exp/qwen1.5b_untied"


def main():
    print("loading base...")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(BASE)

    print("before: tie_word_embeddings =", model.config.tie_word_embeddings)
    emb = model.model.embed_tokens.weight
    lm = model.lm_head.weight
    print(f"embed mean {emb.float().mean().item():.3e} | lm_head mean {lm.float().mean().item():.3e}")
    print("是同一张量吗:", emb.data_ptr() == lm.data_ptr())

    # ★ 关键：untie + lm_head 初始化为 embed 的【拷贝】（等价，不破坏语义）
    model.config.tie_word_embeddings = False
    model.lm_head.weight = torch.nn.Parameter(emb.clone().detach())
    # 同步到 config 里的 generation config（可选）
    model.generation_config.tie_word_embeddings = False

    emb2 = model.model.embed_tokens.weight
    lm2 = model.lm_head.weight
    print("after: tie_word_embeddings =", model.config.tie_word_embeddings)
    print(f"embed mean {emb2.float().mean().item():.3e} | lm_head mean {lm2.float().mean().item():.3e}")
    print("是同一张量吗:", emb2.data_ptr() == lm2.data_ptr())
    print("权重相等吗（应 True）:", torch.allclose(emb2.float(), lm2.float()))

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("SAVED ->", OUT)

    # 冒烟：加载回来检查
    m2 = AutoModelForCausalLM.from_pretrained(OUT, torch_dtype=torch.bfloat16, device_map="cpu")
    print("reload: tie =", m2.config.tie_word_embeddings,
          "| ptr 相同:", m2.model.embed_tokens.weight.data_ptr() == m2.lm_head.weight.data_ptr(),
          "| 数值相等:", torch.allclose(m2.model.embed_tokens.weight.float(), m2.lm_head.weight.float()))
    print("UNTIE BASE READY")


if __name__ == "__main__":
    main()
