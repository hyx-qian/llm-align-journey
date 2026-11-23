# LLM 后训练全流程研究：单轮 RLVR → 多轮工具 Agent

> **Qwen2.5-1.5B-Instruct ｜ 单轮：SFT → RFT → DPO → GRPO ｜ 多轮：冷启动 SFT → 工具调用协议 → 工具生态 ｜ 客观评测 ｜ 全程排障留档**
> 一个 34 天的后训练研究闭环：先从零手写底层，再用框架实战，最后回答"为什么"——
> 完整实验数据见 [`results/RESULTS.md`](results/RESULTS.md)，失败分析见 [`results/FAILURE_ANALYSIS.md`](results/FAILURE_ANALYSIS.md)。

## 🎯 主结果：四级对照实验

| 阶段 | 训练数据 | GSM8K test (1319, 贪心解码) | Δ |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | — | 68.16% (899/1319) | — |
| naive SFT (LoRA r8/r16) | 官方短 CoT（7473 条）× 3ep | 60.58% | **−7.58 侵蚀** |
| RFT-SFT（拒绝采样修复） | 自产判对数据（5463 条）× 3ep | **73.46%** | **+5.30** |
| GRPO（on-policy RLVR 首轮） | 2000 题 × G=8 在线采样（16k rollouts） | 70.28% | +2.12 |
| GRPO v2（串联 RFT 起点续训 + Format reward） | 4000 题 × G=8 × 2ep（64k rollouts，32k/epoch） | **75.21%** | **+7.05** |

**核心发现**：低质 SFT 数据会侵蚀强基座的推理能力——修复不需要新模型，需要更好的数据；在修复后的 RFT checkpoint 上继续 GRPO（串联），可达 **75.21%（+7.05 over base）**。

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

## 📌 v2 路线（已完成 → 75.21%）

> v2 已于 2026-10 完成并评测（4000 题 × G=8 × 2ep，RFT 起点串联 + Format reward）——结果见上方表格与 results/RESULTS.md。以下为当时的路线设计（存档）：

- **串联**：从 RFT-SFT checkpoint 续训 GRPO（离线自蒸馏 + 在线 RLVR 叠加）；
- **难题课程**：53% 组内零方差（数据饥饿）→ 采样池向模型能力边界倾斜；
- **entropy 监控**：继续训练的刹车指标（探索度）。

## 环境

- align：训练 + colocate vLLM rollout（TRL 1.12 / transformers 5.16.1 / peft 0.20.0 / torch 2.11+cu128）
- align-eval：评测专用栈（vLLM 0.20.1 + transformers 4.57.6，跨 run 协议一致性）
- 硬件：RTX 5090 32G（单卡训练 + colocate）

---

## 🔬 v9 双场景工具演化（2026-11 起，进行中）

> 基于导师反馈重设计（原「数学计算」场景的天花板：工具可被心算替代，遗忘测不出）

- **场景 A｜数学推理**：「sympy」（计算层）→ **「+Z3」（SMT 约束推理，心算不可行）**
- **场景 B｜代码**：「code_runner」（运行）→ **「+test_runner」（pytest 通过率）**
- **方法**：文档 / RAFT / DPO / **OPD（黑盒轨迹级，参考 GKD/Lion/ToolAnchor）**
- **指标**：EM / TRA / 成本 + Forget / Discover + **Gap Closure（1.5B 追平 7B 的百分比）**
- **核心问题**：1.5B + 参数级训练 + 重放，能否成本更低、能力逼近 7B？——**取决于工具的不可替代性（调节变量，非筛选器）**

> 旧 T1（calc/python/sympy 计算场景）成果保留为**对照基线**（「results/」+「data/archive_v8/」）。

## 🤖 Agent RL（进行中，2026-11）

> 多轮工具调用 Agent 的完整链路：冷启动 SFT → 可信评测协议 → 工具集演化实验（进行中）
> 代码索引见 [`src/agent_rl/README_代码索引.md`](src/agent_rl/README_代码索引.md)，实验数据见 [`src/agent_rl/results/RESULTS_agent_v1.md`](src/agent_rl/results/RESULTS_agent_v1.md)

### 阶段 0：可信基线（已定稿）

- **冷启动 SFT**：untied base（lm_head = embed.clone()，绕开 peft merge 的 tied 损坏）+ LoRA(r8) + modules_to_save + 多段样本（1433 条，tool_result 移出训练目标）+ 3ep
- **部署**：merge 无 tied 告警 → vLLM 直接加载 merged 模型（无需 LoRA 挂载）；单卡 19G，训练 3.6 分钟
- **基线（50 题，防泄漏尾部题）**：

| 指标 | 值 |
|---|---|
| format_rate（输出 tool_call） | **76%** |
| exec_rate（真实执行 safe_calc） | **76%**（与 format 完全相等，无假调用） |
| 判对率（EM） | **64%**（base 同题 42%，**+22pp**） |
| ans=None | 12% |
| stop_reason | boxed 39 / max_turns 11 |

**协议**：stop=["</tool_call>"] 截断生成 + 真实工具执行 + 结果回填 + 两指标分离 + stop_reason 记录。

### T1 阶段结果（单工具 calc，四条路线对比）

| 实验 | 设置 | n | format | EM | TRA | 成本（输入 tok/题） |
|---|---|---|---|---|
| E1 | 1.5B base + calc 文档（不训练） | 200 | **0%** | 33% | N/A | 344 |
| E1b | **7B base + 同文档（不训练）** | 200 | 45% | **90%** | **99%** | 347 |
| E2 | 1.5B 冷启动 SFT（v9） | 200 | **77%** | **68%** | **82%** | **137** |
| E3c | v9 在「必须 python」题 | 31 | 45% | 35% | **54%** | 104 |

**TRA（结果采纳率）梯度**：7B 99% > 1.5B SFT 82% > 1.5B 跨场景 54%——「调了工具是否真的采纳结果」的能力梯度（Fidelity 恒 1 测不出）。

**Seconds per Correct（秒/正解）**：7B **2.56s**（EM 90%）｜v9 **1.32s**（EM 68%）｜1.5B 文档 1.82s（EM 33%）——**v9 最快（比 7B 快 1.9×）**；每 pp EM 的 token 成本 v9 7.3 vs 7B 16.4：成本敏感选 v9，效果敏感选 7B。


**指标出处（严格标注，不占用他人）**：「Cost-of-Pass」（tokens/EM）引用 2026 的 Cost-of-Pass / Skan 2026 的 cost-per-successful-task；「Tool Precision/Recall/F1」（比「调用了哪些工具」）属 Ragas/Arize/FinMCP-Bench，**本工作的 TRA 是不同层面**（比「工具结果是否被下一轮采纳」，受 RAG faithfulness 启发）；Forget/Discover/AvgPerf 引自 ContDa（ACL 2026）；BWT/FWT 引自 Berkeley 2607；CPRP（cost per recovered point）检索范围内未见，为本工作提出（T2 使用）。

**四个发现**：①文档路线有强规模依赖（1.5B 格式遵循 0% → 7B 44%）②T1 阶段 7B+文档 EM 88% vs 1.5B SFT 68%（+20pp），但推理成本 5-10× ③训练后推理成本反降 60%（文档短） ④「工具必要性」是程度问题（LLM 心算可完成部分），Discover 须用同批题差值（基线 = 35%）。

### 工具生态（T2 进行中）

- 工具：calc（单步算术）/ python（sandbox 代码执行）/ sympy（符号运算）
- 标准工具文档 v1：三工具规格 + 调用格式 + 选择规则（重叠优先 calc、失败回退）+ few-shot（≈682 tokens）
- 研究问题：工具集演化下的新工具适应与旧工具保持（三条件成本-正确率-遗忘对比）
