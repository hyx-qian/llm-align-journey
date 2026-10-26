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
| GRPO r8（首轮） | 在线 RLVR：2000 题 × G=8 × 1 ep（16k rollouts） | **70.28% (927/1319)** | **+2.12** |
| GRPO v2（串联：RFT起点续训 + Format reward） | 4000题 × G=8 × 2ep（64k rollouts 总量，32k/epoch） | **75.21% (992/1319)** | **+7.05** |

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
4. **GRPO 首轮：on-policy RLVR 管线贯通**（base 起点直接强化 +2.12 分）：
   - 训练动态健康：采样判对率 0.66 → 0.775（+12pp 全程上行）、KL 温和
     （0.0002 → 0.003）、无 NaN；
   - 主动刹车：entropy 0.37 → 0.21（探索下降），1 epoch 后即停，防策略坍缩；
   - 数据饥饿先兆：训练末期 53% 的组内 std=0（全对/全错组静音）→ 有效信号
     枯竭，续训需更难题目或加大 G；
   - 未超 RFT（73.46%）的归因：①训练量差 8 倍（250 步 vs 16389 样本次），
     KL 仅 0.003 表明策略远未到位；②起点为 base 而非 RFT checkpoint——
     当前是平级竞争，非串联叠加（见下一步）。

## 下一步（v2 实验设计）

- **串联路线**：从 RFT-SFT checkpoint 续训 GRPO（model 与 π_ref 均为 RFT 模型）
  ——离线自蒸馏与在线 RLVR 叠加，预期突破 73.46%；
- **难题课程**：当前 53% 组静音 → 采样池向模型能力边界题目倾斜，恢复有效信号；
- **entropy 监控**：作为继续训练的刹车指标（探索度）。

## 管线验证记录（0.5B）

- DPO 冒烟：506 对 acc 0.84 / k8 1550 对 acc 0.74、margin +0.54（训练指标）——
  偏好对管线（native-boxed 过滤）验证通过；test 评测待正式实验。

## 复现

数据构造：`src/gsm8k_sft_data.py`（官方 CoT）、`src/rft_data.py`（自采样判对）、
`src/dpo_data.py`（偏好对，native-boxed 过滤）、GRPO 数据内置
`src/grpo_smoke_1_5b.py`（gold_answer 列）
训练：`src/sft_lora.py`（naive r8/r16）、`src/sft_lora_rft.py`（RFT）、
`src/dpo_smoke.py`（DPO 冒烟）、`src/grpo_smoke_1_5b.py`（GRPO 首轮，
vLLM colocate rollout），align 环境（训练 + rollout 同环境）
评测：`src/gsm8k_eval.py` / `src/eval_rft.py`（RFT）/ `src/eval_grpo.py`（GRPO），
align-eval 环境，原生 LoRA 路径
