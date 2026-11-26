# src/agent_rl 代码索引（v9 双场景版，2026-11-23）

> 目录不变（`src/agent_rl`），但文件已按 v9 需求分类：**15 个核心保留 + 9 个 v8 遗留归档 + 6 个待建**。

## 核心（15 个，v9 使用）

| 文件 | 职责 |
|---|---|
| `env_rollout.py` | 多轮 rollout（生成→解析→执行→回填；三工具解析/区间去重/平衡 boxed） |
| `env_reward.py` | 奖励（判对 + 工具有效性） |
| `env_tool_calc.py` | calc（单步算术）——**已内化，不参与演化链**（v9 训练痕迹；E_old 题目用） |
| `env_tool_python.py` | python sandbox——**已部分内化，不参与数学场景**；代码场景 code_runner 底座 |
| `env_tool_sympy.py` | sympy（符号运算）——**数学 T1 的新工具（第 1 次演化）** |
| `env_tool_doc.py` | 三版文档（TOOL_DOC_CALC/T2/T3）——**v9 需加 Z3 版/代码版** |
| `eval_doc_baseline.py` | **主评测**（--doc/--eval/--gpu-mem；记录 TRA/轨迹/latency/tokens） |
| `train_sft.py` | 冷启动 SFT（untied+LoRA+modules_to_save） |
| `train_raft.py` | RAFT 骨架（select_best 待实现） |
| `client_vllm.py` | vLLM 客户端 |
| `analyze_results.py` | 单实验 + 演化指标 + 成本 |
| `analyze_extra_metrics.py` | ExecSR / tok-per-correct / latency |
| `deploy_untie_base.py` | untied base（merge 修复） |
| `data_from_rft.py` | RFT CoT → 工具轨迹（T1 数据链） |
| `data_multiturn.py` | 单段 → 多段（T1 的 1433 条） |

## v9 待建（6 个，学员实现）

| 文件 | 职责 | 状态 |
|---|---|---|
| `env_tool_z3.py` | Z3 约束求解（沙箱：超时 5s / 约束≤200 字符 / 变量≤5） | ⬜ |
| `env_tool_code.py` | code_runner（运行代码，基于 env_tool_python 扩展） | ⬜ |
| `env_tool_test.py` | test_runner（pytest 通过率，subprocess 沙箱） | ⬜ |
| `train_tctd.py` | TCTD（Teacher-Corrected Trajectory Distillation，原 OPD 改名） | ⬜ |
| `data_z3_tasks.py` | Z3 约束题模板化生成（60+ 题） | ⬜ |
| `data_code_tasks.py` | HumanEval/MBPP 加载与格式化 | ⬜ |

## 归档（v8 历史，不影响 v9）

| 位置 | 内容 |
|---|---|
| `data/archive_v8/` | T3 数据（e_new_T3_sympy / e_mix_T3_3tool / e_old_T3_250）+ 旧数据链（gsm8k_coldstart{,_v2,_v3} / e_mix_overlap） |
| `scripts/archive_v8/` | 临时脚本（build_*/mark_*/append_t1）+ **9 个 v8 遗留 py**（data_agent_prompt / data_boundary / data_mix / eval_peft / eval_rollout / eval_smoke / diag_token_acc / test_token_len / data_new_tasks）+ 旧结果 jsonl |

## 数据（8 个 JSONL）

| 文件 | 用途 |
|---|---|
| `e_old_T1_200.jsonl` | T1/T2 的旧题（200） |
| `e_new_python_tasks.jsonl` | python 题（68，含 AIME 30 + 手写 20 + 补 18） |
| `e_new_T2_python_only_30.jsonl` / `e_new_T2_python_only_strict.jsonl` | python 严格子集（30/31） |
| `e_mix_T2_collab.jsonl` | 协作题（31） |
| `gsm8k_agent_train_4k.jsonl` | E_old 来源 |
| `gsm8k_coldstart_v4_multiturn.jsonl` | T1 训练数据（1433） |
| `aime_2024_30.jsonl` | AIME 2024（30） |

## 模型

| 位置 | 内容 |
|---|---|
| `exp/agent_sft_v9_merged/` | v9 基线（merged，vLLM 直接加载） |
| `exp/agent_sft_v9_untied/` | v9 LoRA adapter（可续训） |
| `exp/qwen1.5b_untied/` | untied base |
| `~/models/qwen2.5-7b-instruct/` | 7B（教师/上限参照） |
## 隔离档（档 2）说明

**定义**：每工具一个独立模型（不共享参数）——评测时按题路由（sympy 题用 sympy 模型，z3 题用 z3 模型）。
**实现**：两次独立 SFT（；）——**不需要新脚本**。
**遗忘**：≈0（每个模型只学自己的工具）——**成本 = 模型数 × 存储**（每工具 3.4G）。
**对照**：档 3（全量）= T1 冷启动 → 续训 T2（共享参数）——测真实遗忘。
**评测**：（sympy 题）+ （z3 题）。
