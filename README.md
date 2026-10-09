# GSM8K 后训练全流程研究：侵蚀发现 → 数据修复 → 在线强化

> **Qwen2.5-1.5B-Instruct ｜ SFT → RFT → DPO → GRPO ｜ 客观评测协议 ｜ 全程排障留档**
> 一个 34 天的后训练研究闭环：先从零手写底层，再用框架实战，最后回答"为什么"——
> 完整实验数据见 [`results/RESULTS.md`](results/RESULTS.md)，失败分析见 [`results/FAILURE_ANALYSIS.md`](results/FAILURE_ANALYSIS.md)。

## 🎯 主结果：四级对照实验

| 阶段 | 训练数据 | GSM8K test (1319, 贪心解码) | Δ |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | — | 68.16% (899/1319) | — |
| naive SFT (LoRA r8/r16) | 官方短 CoT（7473 条）× 3ep | 60.58% | **−7.58 侵蚀** |
| RFT-SFT（拒绝采样修复） | 自产判对数据（5463 条）× 3ep | **73.46%** | **+5.30** |
| GRPO（on-policy RLVR 首轮） | 2000 题 × G=8 在线采样（16k rollouts） | 70.28% | +2.12 |

**核心发现**：低质 SFT 数据会侵蚀强基座的推理能力——而修复不需要新模型，需要更好的数据。

## 🔍 三个核心发现

**1. 朴素 SFT 侵蚀强基座（−7.58）**
官方短 CoT（短、浅、风格单一）3 epoch 后系统性退化；badcase 归因：漏约束 /
百分比误读 / 算术错；r8 与 r16 结果完全一致 → 对照实验排除秩因素，定位数据因素。

**2. 数据质量决定 SFT 成败（+5.30 修复）**
训练管线、超参、评测协议全部不变，唯一变量是数据来源——换成本模型自采样 +
可验证 reward 过滤的数据（5463 条）后回升至 73.46%，超过 base。定性发现
**自蒸馏效应**：自采样分布中的正确模式固化为贪心行为。

**3. GRPO 首轮的训练动态（+2.12）**
采样判对率 0.66 → 0.775（+12pp 全程上行）、KL 0.0002 → 0.003 温和、
entropy 0.37 → 0.21 **主动刹车防策略坍缩**、53% 组内零方差（数据饥饿先兆，
定量记录）——on-policy RLVR 管线贯通，undertrained 状态下明确 v2 路线。

## 🧯 排障档案（工程能力证据）

34 天内 5 次重大训练事故，全部**以对照实验定案、全留档**（详见 `src/diagnostics/`）：

- `merge_and_unload()` 权重损坏 → 决定性对照实验（同一 adapter：vLLM 原生加载 69% vs
  合并后 44%）→ 评测全面改走原生 LoRA 路径，机制调查留档
- 评测协议一致性：tokenizer 5.16 序列化格式 vs 4.57 读取端错位 → 权重与 tokenizer
  分离加载
- vLLM 依赖地狱（transformers 被自动降级）→ 训练/推理环境分离 → GRPO colocate
  场景下的受控合并四步 SOP
- 共享服务器资源事件（sshd 假死 / OOM 查案 / swap 风暴）→ 自保三件套流程化

## 📁 仓库结构

```
├── src/                  # 全部管线代码
│   ├── gsm8k_sft_data.py     # 官方 CoT → SFT 格式（\boxed 契约）
│   ├── sft_lora.py           # SFT（r8/r16）
│   ├── rft_data.py           # 拒绝采样数据管线
│   ├── sft_lora_rft.py       # RFT 训练
│   ├── dpo_data.py           # 偏好对构建（native-boxed 过滤）
│   ├── grpo_smoke_1_5b.py    # GRPO 训练（colocate vLLM rollout）
│   ├── gsm8k_eval.py         # 自研评测（抽取+归一化+判分）
│   ├── eval_*.py             # 各模型评测入口
│   └── diagnostics/          # merge bug 诊断实验留档
├── results/              # 评测明细（JSONL）+ RESULTS.md + 失败分析
└── exp/                  # checkpoint（不进 git）
```

## ⚡ 快速复现

```bash
# 1. 数据（官方 CoT → SFT 契约格式）
python src/gsm8k_sft_data.py

# 2. 训练（align 环境：训练 + colocate rollout）
python src/sft_lora.py            # naive SFT r8/r16
python src/sft_lora_rft.py        # RFT
python src/grpo_smoke_1_5b.py     # GRPO（use_vllm=True, colocate）

# 3. 评测（align-eval 环境：vLLM 原生 LoRA，贪心协议一致）
python src/eval_rft.py            # 各模型同理
```

## 📌 v2 路线

- **串联**：从 RFT-SFT checkpoint 续训 GRPO（离线自蒸馏 + 在线 RLVR 叠加）；
- **难题课程**：53% 组内零方差（数据饥饿）→ 采样池向模型能力边界倾斜；
- **entropy 监控**：继续训练的刹车指标（探索度）。

## 环境

- align：训练 + colocate vLLM rollout（TRL 1.12 / transformers 5.16.1 / peft 0.20.0 / torch 2.11+cu128）
- align-eval：评测专用栈（vLLM 0.20.1 + transformers 4.57.6，跨 run 协议一致性）
- 硬件：RTX 5090 32G（单卡训练 + colocate）
