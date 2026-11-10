# -*- coding: utf-8 -*-
"""Agent RL 阶段 1 smoke：50 题多轮 rollout + reward + 统计验证管线端到端。

流程：
  data_agent.py 产出的 agent 数据 → VLLMClient → rollout_multiturn → combined_reward
  → 统计（工具使用率/准确率/轮数分布）→ JSONL 归档

★★★ 统计指标设计由你实现（Agent RL 特有的训练动态信号）：
  1. tool_usage_ratio：num_tool_calls / turns_used 的分布（均值/中位数/分位数）
     ——比率持续偏高=工具依赖，偏低=不会用工具
  2. accuracy_by_tool_use：按"是否使用工具"分组的准确率
     ——直接检验假设：工具调用组的算术错误是否更少
  3. turn_distribution：轮数分布（1轮/2轮/3+轮各占比）
     ——模型倾向一轮直答还是多轮工具协作
先写注释回答"每个指标为什么值得监控"，再写实现。
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 离线用本地缓存（服务器到 HF 不通）
import json
import sys
import os
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client_vllm import VLLMClient
from env_rollout import rollout_multiturn
from env_reward import combined_reward

AGENT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_agent_train_4k.jsonl")
OUT_JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "smoke_agent_v1.jsonl")

N_SMOKE = 50


def main(n_samples: int = N_SMOKE):
    # 加载数据
    rows = []
    with open(AGENT_DATA) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[:n_samples]
    print(f"smoke_agent: {len(rows)} 条")

    # vLLM client（FakeLLM 阶段已验证接口约定）
    client = VLLMClient(model_path="Qwen/Qwen2.5-1.5B-Instruct", gpu_memory_utilization=0.25)

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    results = []
    with open(OUT_JSONL, "w") as f:
        for i, ex in enumerate(rows):
            user_msg = next((m["content"] for m in ex["prompt"] if m["role"] == "user"), "")
            gold = ex.get("gold_answer", "")
            try:
                tr = rollout_multiturn(client, user_msg, sampling_params=None)
                reward = combined_reward(tr, gold)
            except Exception as e:
                print(f"[{i}] ROLLOUT ERROR: {type(e).__name__} {e}")
                continue
            rec = {
                "idx": i,
                "question": user_msg[:80],
                "final_answer": tr.get("final_answer"),
                "num_tool_calls": tr.get("num_tool_calls"),
                "turns_used": tr.get("turns_used"),
                "reward": reward,
                "trajectory_tail": [c[-150:] for _, c in tr.get("trajectory", [])[-3:]],
            }
            results.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rows)} done")

    # ========== 统计指标（★★★ 由你实现） ==========
    # ========== 先写注释回答"每个指标为什么值得监控"，再写实现 ==========
    # 指标1 tool_usage_ratio：num_tool_calls / turns_used 的均值与分布
    #   为什么监控：比率偏高=工具依赖（hacking 风险），偏低=不会用工具（能力问题）
    #   ——这是 Agent RL 特有的训练动态信号（单轮 RLVR 没有这个维度）
    tool_usage_ratio = [rec['num_tool_calls'] / rec['turns_used'] for rec in results]
    tool_usage_summary = {
        "mean": statistics.mean(tool_usage_ratio),
        "median": statistics.median(tool_usage_ratio),
        "p90": sorted(tool_usage_ratio)[int(len(tool_usage_ratio) * 0.9)],
    }
    print("tool_usage_ratio: ", tool_usage_summary)
    
    # 指标2 accuracy_by_tool_use：用工具组 vs 不用工具组的判对率对比
    #   为什么监控：直接检验假设"工具调用消除算术错误类失败"
    #   （判对用 combined_reward 里的 correct 分量：reward >= 1.0 视为答对）
    n_correct = len([rec for rec in results if rec['reward'] > 0])
    n_tool_use = len([rec for rec in results if rec['num_tool_calls'] > 0])
    n_tool_use_acc = len([rec for rec in results if rec['reward'] > 0 and rec['num_tool_calls'] > 0])
    accuracy_by_tool_use = n_tool_use_acc / max(1, n_tool_use)
    accuracy_by_tool_unuse = (n_correct - n_tool_use_acc) / max(1, len(results) - n_tool_use)
    print(f"accuracy_by_tool_use: {accuracy_by_tool_use}, accuracy_by_tool_unuse: {accuracy_by_tool_unuse}")
    
    # 指标3 turn_distribution：1轮/2轮/3+轮的占比
    #   为什么监控：模型倾向一轮直答还是多轮协作——训练前后的变化=策略演化信号
    turn_dist = {"1": 0, "2": 0, "3+": 0}
    for rec in results:
        t = rec["turns_used"]
        bucket = str(t) if t <= 2 else "3+"
        turn_dist[bucket] += 1
    turn_distribution = {k: round(v / len(results), 3) for k, v in turn_dist.items()}
    print("turn_distribution:", turn_distribution)
    
    # raise NotImplementedError("TODO(学员): 三个统计指标")


if __name__ == "__main__":
    import sys as _sys
    n = int(_sys.argv[1]) if len(_sys.argv) > 1 else N_SMOKE
    main(n)
