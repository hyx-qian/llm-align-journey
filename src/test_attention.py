# -*- coding: utf-8 -*-
"""
W1-D2 自动批改器：验证你手写的 attention 是否正确
用法：与 my_attention.py 放同一目录，执行  python test_attention.py
"""
import torch

torch.manual_seed(0)

D, T, B, H = 32, 7, 3, 4   # 隐藏维/序列长/batch/头数


# ============ 标准答案（朴素数学写法） ============
def _ref_single(q, k, v, causal=True):
    att = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    if causal:
        Tq = q.shape[-2]
        mask = torch.triu(torch.ones(Tq, Tq, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
    att = att.softmax(dim=-1)
    return att @ v


def _ref_mha(x, mha, causal=True):
    Bb, Tt, dd = x.shape
    Hh, hd = mha.n_heads, mha.d_head
    q = mha.Wq(x).view(Bb, Tt, Hh, hd).transpose(1, 2)
    k = mha.Wk(x).view(Bb, Tt, Hh, hd).transpose(1, 2)
    v = mha.Wv(x).view(Bb, Tt, Hh, hd).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(Tt, Tt, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
    att = att.softmax(dim=-1)
    out = (att @ v).transpose(1, 2).contiguous().view(Bb, Tt, dd)
    return mha.Wo(out)


RESULTS = []

def report(name, ok, hint=""):
    RESULTS.append(ok)
    print(("✅ PASS  " if ok else "❌ FAIL  ") + name)
    if not ok and hint:
        print(f"       ↳ 诊断：{hint}")


def main():
    try:
        from my_attention import MultiHeadAttention, single_head_attention
    except Exception as e:
        print(f"❌ 无法从 my_attention 导入：{e}")
        print("   检查两个名字拼写：single_head_attention / MultiHeadAttention")
        return

    try:
        q, k, v = (torch.randn(B, T, D) for _ in range(3))
        got = single_head_attention(q, k, v, causal=True)
        exp = _ref_single(q, k, v, causal=True)
        report("T1 单头(causal) 数值正确",
               got.shape == exp.shape and torch.allclose(got, exp, atol=1e-5),
               "逐项检查：①分数是否除以 sqrt(d_head)；②mask 是否 triu(diagonal=1) 屏蔽未来；③softmax 是否在 dim=-1。")
    except Exception as e:
        report("T1 单头(causal) 数值正确", False, f"抛异常：{type(e).__name__}: {e}")

    try:
        got = single_head_attention(q, k, v, causal=False)
        exp = _ref_single(q, k, v, causal=False)
        report("T2 单头(无mask) 数值正确", torch.allclose(got, exp, atol=1e-5),
               "causal=False 时不应有任何 masked_fill。")
    except Exception as e:
        report("T2 单头(无mask) 数值正确", False, f"抛异常：{type(e).__name__}: {e}")

    try:
        torch.manual_seed(1)
        mha = MultiHeadAttention(d_model=D, n_heads=H)
        x = torch.randn(B, T, D)
        out = mha(x, causal=True)
        report("T3 MultiHead 输出形状 (B,T,d)",
               tuple(out.shape) == (B, T, D),
               f"期望 {(B, T, D)}，得到 {tuple(out.shape)} —— 检查拆头/拼回的维度顺序。")
    except Exception as e:
        report("T3 MultiHead 输出形状", False, f"抛异常：{type(e).__name__}: {e}")

    try:
        torch.manual_seed(1)
        mha = MultiHeadAttention(d_model=D, n_heads=H)
        torch.manual_seed(2)
        x = torch.randn(B, T, D)
        got = mha(x, causal=True)
        exp = _ref_mha(x, mha, causal=True)
        report("T4 MultiHead(causal) 数值正确", torch.allclose(got, exp, atol=1e-5),
               "常见坑：①忘了除 sqrt(d_head)；②view/transpose 顺序错（先 view(B,T,H,hd) 再 transpose(1,2)）；③拼回时忘了 contiguous；④忘了过 Wo。")
    except Exception as e:
        report("T4 MultiHead(causal) 数值正确", False, f"抛异常：{type(e).__name__}: {e}")

    try:
        torch.manual_seed(1)
        mha = MultiHeadAttention(d_model=D, n_heads=H)
        torch.manual_seed(3)
        x1 = torch.randn(B, T, D)
        x2 = x1.clone()
        x2[:, -1] += 5.0                     # 只扰动最后一个 token
        o1 = mha(x1, causal=True)
        o2 = mha(x2, causal=True)
        report("T5 因果性：改动未来不改变过去输出",
               torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-5),
               "你的 mask 方向反了：应把 j>i 的位置（上三角，triu diagonal=1）置 -inf。")
    except Exception as e:
        report("T5 因果性", False, f"抛异常：{type(e).__name__}: {e}")

    try:
        torch.manual_seed(1)
        mha = MultiHeadAttention(d_model=D, n_heads=H)
        torch.manual_seed(2)
        x = torch.randn(B, T, D)
        got = mha(x, causal=False)
        exp = _ref_mha(x, mha, causal=False)
        report("T6 MultiHead(无mask) 数值正确", torch.allclose(got, exp, atol=1e-5),
               "causal=False 分支是否真的跳过了 mask？")
    except Exception as e:
        report("T6 MultiHead(无mask) 数值正确", False, f"抛异常：{type(e).__name__}: {e}")

    n = sum(RESULTS)
    print(f"\n========== 得分：{n}/{len(RESULTS)} ==========")
    if n == len(RESULTS):
        print("🎉 全部通关！执行 git add . && git commit，然后回来找我：")
        print("   1) 口头复述 5 道自测题答案")
        print("   2) 我们进 Day3：手写 IMDB 训练循环")


if __name__ == "__main__":
    main()
