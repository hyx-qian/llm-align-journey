# GSM8K 后训练实验结果

> **评测协议**：GSM8K test 全量 1319 条，vLLM 贪心解码（temperature=0, max_tokens=512），
> `\boxed{}` 抽取 + 归一化判分（`src/gsm8k_eval.py`）。所有 LoRA 模型走 vLLM 原生加载
> ——`merge_and_unload()` 在 transformers 5.16.1 + peft 0.20.0 组合下产出的合并权重有损
> （同一 adapter：原生加载 69% vs 合并后 44%，诊断见 `src/diagnostics/`，机制待查）。

## 主结果

| 模型 | 训练数据 | GSM8K test 准确率 | 相对 base |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | — | 68.16% (899/1319) | — |
| naive SFT r8 | 官方短 CoT（7473 条）× 3 ep | 60.58% (799/1319) | **−7.58** |
| naive SFT r16 | 同上 | 60.58% (799/1319) | −7.58 |
| RFT-SFT r8 | base 自采样 temp=1.0 判对数据（5463 条）× 3 ep | **73.46% (969/1319)** | **+5.30** |

## 发现

1. **朴素 SFT 侵蚀强基座**：官方短 CoT（短、浅、风格单一）训 3 epoch 后，准确率
   68.16% → 60.58%；错误样本归因：漏约束 / 百分比误读 / 算术错。r8 与 r16 完全
   一致 → 秩不是瓶颈，指向数据因素。
2. **数据质量决定 SFT 成败**：训练管线、超参（r=8, lr=1e-4, 有效 batch=32）、评测
   协议全部不变，唯一变量是数据来源——换成本模型自采样 + 正确性过滤的数据后，
   准确率回升至 73.46%，超过 base 5.30 分。（注：RFT 数据量为判对覆盖的自然结果，
   覆盖 73% 的题目；数据多样性增强实验在 backlog。）
3. **自蒸馏效应**：RFT 将模型自采样分布中"隐含的正确模式"固化为贪心行为——
   模型自己能采出来的解法，教给贪心解码复现。

## 进行中

- DPO（0.5B 冒烟）：506 对 acc 0.84 / k8 1550 对 acc 0.74、margin +0.54（训练指标）；
  test 评测待正式实验，当前定位为管线验证 + 偏好对数据构建验证（native-boxed 过滤）。

## 复现

数据构造：`src/gsm8k_sft_data.py`（官方 CoT）、`src/rft_data.py`（自采样判对）、
`src/dpo_data.py`（偏好对，native-boxed 过滤）
训练：`src/sft_lora.py`（naive r8/r16）、`src/sft_lora_rft.py`（RFT）、`src/dpo_smoke.py`（DPO 冒烟），align 环境
评测：`src/gsm8k_eval.py` / `src/eval_rft.py`，align-eval 环境，原生 LoRA 路径
