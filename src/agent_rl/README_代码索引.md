# src/agent_rl 代码索引

> 2026-11-10 重构后（方案 C：按"功能域_对象"命名）。所有文件通过 `env_*` / `data_*` / `train_*` / `eval_*` / `deploy_*` / `diag_*` 前缀可一眼看出职责。

## 环境与工具（env_*）

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `env_tool_calc.py` | 计算器工具（AST 白名单 safe_calc） | `safe_calc(expr) -> str`（异常→"ERROR: ..."） |
| `env_tool_python.py` | Python 代码执行器（sandbox：AST 检查/受限 builtins/SIGALRM 超时 5s/输出截断） | `safe_python(code, timeout=5) -> str` |
| `env_tool_sympy.py` | 符号运算（solve/simplify/expand/factor/diff/integrate） | `safe_sympy(expr, op="solve", var="x") -> str` |
| `env_tool_doc.py` | 标准工具文档 v1（三工具规格+调用格式+选择规则+few-shot，≈682 tokens） | `TOOL_DOC_V1` 常量（B/C 文档路线共用） |
| `env_rollout.py` | 多轮 rollout 环境（生成→解析→执行→回填→继续；stop 兼容正则） | `rollout_multiturn(client, prompt, sp, max_turns=5)`；`TOOL_SYSTEM_PROMPT`；`TOOL_CALL_RE`；`BOXED_RE` |
| `env_reward.py` | 奖励函数（判对 + 工具有效性比率，None 防御） | `combined_reward(rollout, gold) -> float` |

## 数据构造（data_*）

| 文件 | 职责 | 产物 |
|---|---|---|
| `data_from_rft.py` | v2 RFT CoT → 工具调用轨迹（LaTeX 转纯文本 + safe_calc 重放验证） | `data/gsm8k_coldstart.jsonl` |
| `data_multiturn.py` | 单段 → 多段样本（tool_result 移出训练目标） | `data/gsm8k_coldstart_v4_multiturn.jsonl`（1433 条） |
| `data_boundary.py` | 边界示范（A 完成即停 / B ERROR 恢复 / C 语言过滤）+ v2 混合 | `data/gsm8k_coldstart_v2.jsonl`（285 条） |
| `data_mix.py` | v3 配比（工具示范 ×3 上采样 + 纯 CoT 50，seed=42） | `data/gsm8k_coldstart_v3.jsonl`（455 条） |
| `data_agent_prompt.py` | v2 数据加工具说明 system prompt | `data/gsm8k_agent_train_4k.jsonl` |

## 训练（train_*）

| 文件 | 职责 | 说明 |
|---|---|---|
| `train_sft.py` | 冷启动 SFT（**当前配置：untied base + LoRA + modules_to_save + 多段数据 + 3ep**） | 产出 `exp/agent_sft_v9_untied/`；单卡 19G，3.6 分钟 |
| `train_raft.py` | 迭代式拒绝采样微调（RAFT）骨架 | `select_best` 待实现；`ITERATIVE_MODE` 参考块 |

## 评测（eval_*）

| 文件 | 职责 | 说明 |
|---|---|---|
| **`eval_rollout.py`** | **主评测**：完整多轮 rollout + 四指标（format / exec / EM / ans=None）+ stop_reason | 协议：`stop=["</tool_call>"]` + 真实执行 + `max_turns=5` |
| `eval_peft.py` | transformers+PEFT 推理对照（不 merge） | 用于诊断 merge/加载一致性 |
| `eval_coldstart_legacy.py` | 旧版验证（vLLM+LoRA adapter，历史版本） | 归档用，不再更新 |
| `eval_smoke.py` | 冒烟（50 题快速 + 指标统计） | 快速回归用 |

## 部署与诊断（deploy_* / diag_* / test_*）

| 文件 | 职责 |
|---|---|
| `deploy_untie_base.py` | 生成 untied base（`tie_word_embeddings=False` + `lm_head.weight = embed.clone()`），解决 merge tied 问题 |
| `diag_token_acc.py` | 分位置 token accuracy 诊断（定位 special token 学习问题） |
| `test_token_len.py` | token 长度分布（选 max_length 的依据） |
| `client_vllm.py` | vLLM 客户端（`VLLMClient` / `VLLMLoRAClient`） |

## 实验产物（exp/）

| 目录 | 内容 |
|---|---|
| `qwen1.5b_untied/` | untied base（merge 无 tied 问题的基础） |
| `agent_sft_v9_untied/` | v9 LoRA checkpoint（阶段 0 基线训练产物） |
| `agent_sft_v9_merged/` | v9 merged 完整模型（**vLLM 直接加载**） |
| `agent_sft_v8_full/` | v8 全参 checkpoint（32G 高成本对照，未验证） |
| `agent_sft_coldstart_v4/` | v4 LoRA（PEFT 推理 50% 的历史版本） |

## 快速上手（三条命令）

```bash
# 1. 主评测（需 GPU）：50 题完整多轮
python eval_rollout.py

# 2. 冷启动训练（需 GPU，~4 分钟）
python train_sft.py

# 3. 生成 untied base（一次性）
python deploy_untie_base.py
```
