# Agent RL 阶段实验结果汇总（Agent RL 阶段 1/1.5）

> 2026-11-04 ｜ 覆盖：Agent RL 阶段 1（多轮工具调用）+ SFT 冷启动 v1-v4 + 部署排查 ｜
> 服务器数据目录：src/agent_rl/ ｜ 训练/评测环境：align（miniforge3/envs/align，vllm 0.20.1 / transformers 5.16.1 / peft 0.20.0 / trl 1.12.0）
> 与主 RESULTS.md 的关系：主文档覆盖单轮 RLVR 管线（v1/v2/75.21%），本文档只记 **Agent 多轮线**。

---

## 一、实验结果总表（Agent RL 阶段 1）

| 模型/阶段 | 训练数据 | 训练配置 | 工具调用率 | 判对率 | ans=None 率 | 备注 |
|---|---|---|---|---|---|---|
| **base**（Qwen2.5-1.5B-Instruct） | — | — | **0%**（50/50） | 42%（21/50） | — | 工具调用 zero-shot 遵循率 0%（发现 7） |
| **SFT 冷启动 v1** | 200 条工具示范（RFT CoT 改写） | LoRA r8, **10ep**, lr 1e-4 | **44%**（22/50） | **12%**（6/50）↓ | **36%** | 遗忘：非工具组判对 42%→**7%**；失败模式 X/Y/Z |
| **SFT 冷启动 v2** | 285 条（135 工具 47% + 150 纯CoT 53%，防遗忘混入） | LoRA r8, **3ep**（防过拟合） | **0%**（0/50） | **72%**（36/50）↑ | — | **配比稀释**：纯 CoT 53% → 模型学成"工具可选"→0% 调用 |
| **SFT 冷启动 v3** | 455 条（405 工具 **89%** + 50 纯CoT，×3 上采样 seed42） | LoRA r8, 3ep | **0-2%**（0-1/50） | **68-70%**（34-35/50） | — | **配比证伪**：8:1 强配比仍不调用 → **配比不是根因** |
| **SFT 冷启动 v4（★当前最优）** | 同 v3 + **modules_to_save=[embed_tokens, lm_head]** | LoRA r8 + **embedding/lm_head 解冻**, 3ep | **50%**（5/10，PEFT 推理） | **70%**（7/10） | — | **根因修复**：lm_head 冻结锁死 special token 生成 |

### 关键数字链（工具调用率演化）

```
base 0% → v1 44%（10ep 过拟合+遗忘）→ v2 0%（配比稀释）→ v3 0-2%（8:1 证伪）→ v4 50%（lm_head 解冻）
```

### 各版本诊断要点

| 版本 | 验证方式 | 关键发现 |
|---|---|---|
| v1 | verify_coldstart.py（vLLM+LoRA adapter） | 失败模式 X（失控调用 tool=9）/Y（韩语希伯来语幻觉）/Z（HTML 塞 calc→ERROR） |
| v2 | verify_coldstart_v2（绝对路径修复后） | 纯 CoT 路径健康（72%）但工具路径 0%——配比稀释 |
| v3 | verify_coldstart_v3（学员跑） | 8:1 强配比仍 0-2%——配比证伪 → 指向更深根因 |
| v4 | **verify_peft.py（transformers+PEFT 推理）** | 工具调用 50% ✓——**根因：LoRA target_modules 锁死 special token 生成** |

---

## 二、深度诊断记录（分位置 token accuracy——根因定位）

**方法**：teacher-forcing 冷启动示范轨迹，统计 completion 各位置的 next-token 预测准确率。

| 位置 | token 数 | 准确率 |
|---|---|---|
| `<tool_call>` 标记处 | 2 | **0.0%** |
| 其他位置（推理文字） | 199 | **88.4%** |

**结论**：v3 LoRA SFT **没有改变 `<tool_call>` 位置的生成分布**（teacher-forcing 全错）——90.8% 平均准确率被推理文字（模型已会的 80%+ 部分）稀释。
**机制**：LoRA target_modules=["q_proj","k_proj","v_proj","o_proj"] **不包含 lm_head/embed_tokens**——special token 151657 的生成 logit 被 base 冻结——LoRA 低秩更新不足以抬升。
**修复**：`modules_to_save=["embed_tokens","lm_head"]`（这两层全参数可训练）→ v4 工具调用 50% ✓——**根因链闭环**。

**发现 A（训练分布内生成）**：贪心解码训练分布内 prompt，模型输出纯文本心算（LaTeX）+ **自己幻觉 `<tool_result>CALCULATE(48+24)</tool_result>`**（无真实 `<tool_call>`，自编大写变体）——形式学到一点、顺序完全错误。

---

## 三、部署排查记录（vLLM × LoRA × merge 的三条边界）

| 问题 | 现象 | 根因 | 解法/状态 |
|---|---|---|---|
| vLLM LoRA 加载相对路径 | `LoRAAdapterNotFoundError`（被当 HF repo 在线下载） | vLLM `get_adapter_absolute_path` 非绝对路径 → 下载分支 | **绝对路径** ✓ |
| **vLLM LoRA 不支持 modules_to_save** | `ValueError: vLLM only supports modules_to_save being None` | vLLM LoRA 推理只支持标准低秩 adapter | **merge 后推理**（当前解法）/ PEFT 推理（慢但可靠） |
| **merge_and_unload 产物损坏** | PEFT 推理 50% 工具调用 vs merged（transformers 加载）**0%**、merged vLLM **0%** | **peft 0.20.0 tied embeddings（tie_word_embeddings=True）+ modules_to_save=[embed_tokens, lm_head] 的 merge 权重丢失**（merge 警告："embeddings no longer tied"，peft issue 2777 已知 complications） | **untie 重训**（tie_word_embeddings=False，一行 config，2 分钟）——待执行 |
| HF 加载卡死 | `Errno 101 Network is unreachable` 重试 7 分钟 | hub id 加载在线检查 + 服务器到 HF/hf-mirror 断（**12 小时内网络变化**：昨天 429 可达） | `HF_HUB_OFFLINE=1` 固化脚本 + **BASE 用本地 snapshot 绝对路径** |
| 快照不完整 | `IncompleteSnapshotError`（缺 .gitattributes/LICENSE/README.md） | 离线严格完整性检查 | BASE 指向本地 snapshot 目录（跳过检查） |

**merge bug 证据链（开源 issue 素材，peft 0.20.0 / transformers 5.16.1）**：
1. PEFT 推理（base + adapter 动态挂载）→ 50% 工具调用 ✓
2. merge_and_unload → safetensors → transformers 加载 → **0%** ❌
3. merge_and_unload → safetensors → vLLM 加载 → **0%** ❌
→ merge 环节权重丢失（tied embeddings 解绑 + modules_to_save 双层，警告日志在案）。

---

## 四、与单轮 RLVR 主线的关系（避免数据混淆）

| 线 | 环境 | 评测 | 数字 |
|---|---|---|---|
| 单轮 RLVR 主线（v1/v2） | align 训练 / **align-eval 评测** | gsm8k_eval.py（1319 test） | 68.16 / 60.58 / 73.46 / 75.21 |
| **Agent 多轮线（本文档）** | **align 全程**（train-serve 一致） | rollout_multiturn + combined_reward（同一批题） | 见第一节表 |

- **环境原则**：Agent RL 的"评测"（rollout 采样+reward）是**训练管线的一部分**（train-serve 一致）→ 用 align ✓ 正确；
- **衍生校准任务**：与历史数字对比时需环境校准（base 在 align 下复测 vs align-eval 时代数字）——待并入下次 smoke。

---

## 五、发现档案（FAILURE_ANALYSIS Agent 线增补）

| # | 发现 | 证据 | 对应修复 |
|---|---|---|---|
| 7 | base 对自定义工具格式 zero-shot 遵循率 **0%**（50/50） | smoke v1 全量 | SFT 冷启动（R1 式 cold start） |
| 8 | **灾难性遗忘**：10ep×200 条 → 非工具组判对 42%→7% | verify v1 50 题 | 混入纯 CoT + epoch 10→3 |
| 9 | **衔接缺失**：ans=None 率 36%（工具后不收敛到 boxed） | verify v1 轨迹分析 | 衔接示范（待 v4 全量验证） |
| 10 | **配比不是根因**：8:1 强配比仍 0-2% 调用 | v3 验证 | 深度诊断 → lm_head 根因 |
| 11 | **LoRA target_modules 锁死 special token 生成**（tool_call 处 token acc 0% vs 其他 88.4%） | 分位置诊断 | modules_to_save → 50% ✓ |
| 12 | **merge_and_unload 产物损坏**（tied+modules_to_save 组合） | 三加载器对照实验 | untie 重训（待执行）/ 开源 issue |
| 13 | 采样发散：语言混杂（韩/希伯来）、HTML 塞 calc | 轨迹尾部 | 温度 0.7 + 语言过滤 + ERROR 恢复示范 |

---

## 六、下一步（按 PLAN_v3 顺序）

| # | 任务 | 前置 | 验收 |
|---|---|---|---|
| 1 | **untie 重训**（sft_agent.py 加 tie_word_embeddings=False，2 分钟） | 报备 | merge 后工具调用率 = PEFT 水平（~50%）→ **merge 归因闭环** |
| 2 | select_best 实现（组内筛选，参考代码已给） | — | 三思考点注释+实现 |
| 3 | 迭代式 RAFT 3 轮（动态数据策略 if-then 随轮） | untie merge 模型 | reward_history 递增 + 工具率 44%→60%+ |
| 4 | 50 题验收（M1 收官） | 迭代完成 | 工具率/ans=None/判对三指标 |
| 5 | 阶段 2 WebShop（PLAN_v3 阶段 2 七天计划） | M1 | 成功率 ≥20% |
