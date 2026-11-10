# -*- coding: utf-8 -*-
"""PLAN_v8 收尾更新：①实验清单回填 T1 状态（E1/E1b/E2/E3c）②前置工作完成项
③指标补 Discover 基线 ④执行顺序改 T2 计划 ⑤合并旧纪律节。"""

PL = "/home/huyx/study/projects/llm-align-journey/PLAN_v8_完备实验规划.md"
s = open(PL, encoding="utf-8").read()

# ---------- ① 实验清单：T1 回填 ----------
old_t1 = """| **E1** | T1-文档 | base(1.5B) | 只给 calc 文档，**不训练** | E_old(200) | **不训练、光靠文档，会用 calc 吗？** | ⬜ 今天跑 |
| **E2** | T1-训练 | base | 冷启动 SFT（1433 条 calc 示范） | E_old(200) | 训练后会用吗？ | ✅ v9 已训（fmt76/exec76/EM64@50）；补 200 |
| **E3** | T1-python参照 | v9 | 不训练，直接考 python 题 | E_new(50) | v9 会不会 python？（**应≈0**，作为 FWT 起点） | ⬜ 今天跑 |"""
new_t1 = """| **E1** | T1-文档1.5B | base(1.5B) | 只给 calc 文档，不训练 | E_old(200) | 不训练光靠文档会用 calc 吗？ | ✅ **fmt 0% / EM 33% / 344 tok** |
| **E1b** | T1-文档7B | base(7B) | 同文档 | E_old(50) | 大模型文档够不够？ | ✅ **fmt 44% / EM 88% / 347 tok** |
| **E2** | T1-训练 | base | 冷启动 SFT（1433 条 calc 示范） | E_old(200) | 训练后会用吗？ | ✅ **fmt 77% / EM 68% / 137 tok** |
| **E3c** | T1-python基线 | v9 | 不训练，考 python_only(31) | E_new_T2_strict(31) | v9 会不会 python？ | ✅ **EM 35%（Discover 基线）** |"""
if old_t1 in s:
    s = s.replace(old_t1, new_t1)
    print("① 实验清单 T1 已回填")

# ---------- ② 前置工作状态 ----------
old_pre = """| 1 | **env_rollout 三工具** | calc/python/sympy 解析+分发+区间去重 | ✅ 完成（8 测试过） |
| 2 | **三工具实现** | calc（已有）/python sandbox/sympy | ✅ 完成（单测过） |
| 3 | **TOOL_DOC_V1** | 三工具文档（规格+规则+few-shot，682 tokens） | ✅ 完成 |"""
new_pre = """| 1 | **env_rollout 三工具** | calc/python/sympy 解析+分发+区间去重 | ✅ 完成（8 测试过） |
| 2 | **三工具实现** | calc（已有）/python sandbox/sympy | ✅ 完成（单测过） |
| 3 | **三版文档** | TOOL_DOC_CALC(268 tok)/T2(496)/T3(563) | ✅ 完成（随阶段递增） |
| 3b | **评测集** | E_old_T1(200)/E_old_T3(250)/E_new_T2(68)/E_mix_T2(31)/E_new_T3(30)/E_mix_T3(33) + **strictness 三档** | ✅ 完成（594 题） |
| 3c | **分析脚本** | analyze_results.py（单实验+演化+成本） | ✅ 完成 |"""
if old_pre in s:
    s = s.replace(old_pre, new_pre)
    print("② 前置工作已更新")

# ---------- ③ 指标补 Discover 基线 ----------
old_d = "| Discover | `EM_Enew(t)` | ContDa |"
new_d = "| Discover | `EM_Enew(t) − EM_E3c`（**基线 = E3c 的 35%**，非 0——LLM 心算让\"必须工具\"模糊） | ContDa + 我们的修正 |"
if old_d in s:
    s = s.replace(old_d, new_d)
    print("③ Discover 基线已修正")

# ---------- ④ 执行顺序改 T2 ----------
old_sched = s[s.index("## 七、执行顺序"):s.index("## 八、风险与备选")]
new_sched = """## 七、执行顺序（T1 已完成 → T2 计划）

**T1（2026-11-10 完成）**：E1 / E1b / E2 / E3c 四实验 + 594 题评测集 + 三版文档 + 分析脚本。

**T2（下一步，按依赖排序）**：

| 天 | 任务 | 产出 |
|---|---|---|
| 11.11 | **python_demos 训练数据 300 条**（构造 + safe_python 重放验证） | T2 的 A1/A2 前置 |
| 11.11 | **E4（1.5B + T2 文档）/ E5（7B + 同文档）** | 文档路线 T2 数据 |
| 11.12 | **E6（RAFT）**：v9 采样 → 筛 → SFT（calc+python 数据） | 参数路线 1 |
| 11.13 | **E7（DPO）**：偏好对构造 + DPOTrainer | 参数路线 2（RL 算法档） |
| 11.14 | **T2 横评**（Forget/Discover/BWT/成本）+ 消融 E12/E13 | **T2 结论（论文核心）** |
| 11.15+ | T3（sympy）——时间紧可延后 | 累积遗忘 |

"""
s = s.replace(old_sched, new_sched)
print("④ 执行顺序已改为 T2 计划")

# ---------- ⑤ 合并旧纪律节 ----------
old_dis = s[s.index("## 十、纪律（不变）"):s.index("## 十一、报备与实验纪律")]
new_dis = """## 十、纪律（合并见第十一节）

见「十一、报备与实验纪律」。

"""
s = s.replace(old_dis, new_dis)
print("⑤ 旧纪律节已合并")

open(PL, "w", encoding="utf-8").write(s)
print("\nPLAN 更新完成，总行数:", len(s.split("\n")))
