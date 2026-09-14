# -*- coding: utf-8 -*-
"""
W1-D2 作业：从零手写 Attention
规则：只用基础算子（@、softmax、view、transpose、masked_fill 等），
     禁止 F.scaled_dot_product_attention / nn.MultiheadAttention。
写完运行：python test_attention.py  —— 6/6 即通关
"""
import math
import torch
import torch.nn as nn


# ===== 第一步：单头注意力（先拿下它，多头只是它的分头批量版） =====
def single_head_attention(q, k, v, causal=True):
    """
    q, k, v: (B, T, D)  —— 已经过投影
    返回:     (B, T, D)
    1) 分数  att = q @ k 转置 / sqrt(D)         -> (B, T, T)
    2) causal 时把"未来"位置填 -inf             -> torch.triu(..., diagonal=1) 生成 mask
    3) attn 权重 = softmax(att, dim=-1)
    4) 输出 = 权重 @ v
    """
    # TODO-1 -- 删掉下面这行，开始写你的代码
    
    # raise NotImplementedError("TODO-1: 实现单头注意力")
    attn = q @ k.transpose(-2, -1) / math.sqrt(k.shape[2]) # (B, T, T)
    if causal:
        mask = torch.triu(torch.ones(attn.shape[1], attn.shape[1], device=attn.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
    attn = attn.softmax(dim=-1)
    return attn @ v # (B, T, D)


# ===== 第二步：多头注意力 =====
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # 属性名必须与批改器一致（批改器直接读这些权重当标准答案）
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        """
        x: (B, T, d_model) -> 返回 (B, T, d_model)
        流程：
          (1) x 分别过 Wq/Wk/Wv                     -> 三个 (B,T,d)
          (2) 拆头: view(B,T,H,hd).transpose(1,2)    -> (B,H,T,hd)
          (3) 每个头调用 single_head_attention（允许用 for 循环逐头算）
          (4) 拼回: transpose(1,2).contiguous().view(B,T,d)
          (5) 过 Wo
        """
        # TODO-2 -- 删掉下面这行，开始写你的代码
        # raise NotImplementedError("TODO-2: 实现多头注意力")
        B, T, D = x.shape
        H, hd = self.n_heads, self.d_head
        q = self.Wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.Wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.Wv(x).view(B, T, H, hd).transpose(1, 2)
        tmp = []
        for i in range(H):
            tmp.append(single_head_attention(q[:,i,:,:], k[:,i,:,:], v[:,i,:,:], causal).unsqueeze(dim=1))
        attn = torch.cat(tmp, dim=1).transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(attn)
        
            
        
        
