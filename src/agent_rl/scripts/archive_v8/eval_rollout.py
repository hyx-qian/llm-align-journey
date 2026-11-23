# -*- coding: utf-8 -*-
"""v9 完整多轮 rollout 验证（阶段 0 正式版）。

修复与新增：
  ① 用 rollout_multiturn（内部 TOOL_SYSTEM_PROMPT，与训练完全一致）
  ② SamplingParams(stop=["</tool_call>"])：生成到第一个调用结束即停（不在同轮自编结果/boxed）
  ③ 两指标分离：format_rate（轨迹含 <tool_call>）/ exec_rate（真实执行 > 0）
  ④ stop_reason 记录（tool_call / boxed / max_turns / no_output）
  ⑤ 真实工具执行（safe_calc）+ 结果回填 + 继续 —— 完整多轮
  ⑥ 判分复用 gsm8k_eval.score（与主线协议一致）
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import LLM, SamplingParams
from env_rollout import rollout_multiturn
from client_vllm import VLLMClient
from gsm8k_eval import score

MODEL = "exp/agent_sft_v9_merged"          # untied base + LoRA(merge) 的完整模型
AGENT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k_agent_train_4k.jsonl")
N_TEST = 50
OUT = "results/verify_rollout_v9.jsonl"


def main():
    client = VLLMClient(MODEL, gpu_memory_utilization=0.35)
    # ★ stop：生成到第一个 </tool_call> 即停
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512, stop=["</tool_call>"])

    rows = [json.loads(l) for l in open(AGENT_DATA) if l.strip()][-N_TEST:]
    results = []
    n_fmt = n_exec = n_correct = n_none = 0
    reasons = {}

    for i, ex in enumerate(rows):
        user_msg = next((m["content"] for m in ex["prompt"] if m["role"] == "user"), "")
        gold = ex.get("gold_answer", "")
        try:
            tr = rollout_multiturn(client, user_msg, sp)
        except Exception as e:
            print(f"[{i}] ERROR {type(e).__name__}: {e}")
            continue
        traj_text = " ".join(c for _, c in tr["trajectory"])
        has_fmt = "<tool_call>" in traj_text
        has_exec = tr["num_tool_calls"] > 0
        ans = tr["final_answer"]
        em = score(ans, gold) if ans else 0.0
        # stop_reason 推断
        if tr["turns_used"] >= 5:
            reason = "max_turns"
        elif ans is not None:
            reason = "boxed"
        elif has_exec:
            reason = "tool_call"
        else:
            reason = "no_output"
        reasons[reason] = reasons.get(reason, 0) + 1
        n_fmt += int(has_fmt); n_exec += int(has_exec)
        n_correct += int(em >= 1.0); n_none += int(ans is None)
        results.append({"idx": i, "turns": tr["turns_used"], "tool_calls": tr["num_tool_calls"],
                        "final_answer": ans, "gold": gold, "em": em,
                        "format": has_fmt, "exec": has_exec, "stop_reason": reason})
        print(f"[{i}] turns={tr['turns_used']} calls={tr['num_tool_calls']} "
              f"fmt={has_fmt} exec={has_exec} ans={ans} gold={gold} em={em:.0f} reason={reason}")

    n = len(results)
    print("=" * 56)
    print(f"format_rate（轨迹含 <tool_call>）: {n_fmt}/{n} = {n_fmt/max(1,n)*100:.0f}%")
    print(f"exec_rate（真实执行 > 0）:        {n_exec}/{n} = {n_exec/max(1,n)*100:.0f}%")
    print(f"判对率（EM）:                    {n_correct}/{n} = {n_correct/max(1,n)*100:.0f}%")
    print(f"ans=None 率:                     {n_none}/{n} = {n_none/max(1,n)*100:.0f}%")
    print(f"stop_reason 分布: {reasons}")

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
