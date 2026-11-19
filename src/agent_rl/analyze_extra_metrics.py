# -*- coding: utf-8 -*-
"""从已有结果 JSONL（含 trajectory）补算扩展指标：

  Call Precision    = 1 − 含 ERROR 的工具调用数 / 总调用数
  Turn Efficiency   = 平均 assistant 轮数
  Cost per Correct  = 平均 (in_tok+out_tok) / EM
  Avg Latency       = 平均每题耗时（秒）
"""
import glob
import json
import os

RESULTS = "results"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def extra_metrics(rows):
    n = len(rows)
    if not n:
        return None
    total_calls = error_calls = 0
    turns, toks, lats, tra = [], [], [], []
    em = 0
    for r in rows:
        traj = r.get("trajectory") or []
        if traj:
            turns.append(sum(1 for role, _ in traj if role == "assistant"))
            for role, content in traj:
                if role != "tool":
                    continue
                total_calls += 1
                if "ERROR" in str(content):
                    error_calls += 1
        else:
            if "turns" in r:
                turns.append(r["turns"])
            total_calls += r.get("calls", 0)
        toks.append((r.get("in_tok", 0) or 0) + (r.get("out_tok", 0) or 0))
        if float(r.get("em", 0)) >= 1.0:
            em += 1
        if r.get("tra") is not None and r.get("calls", 0) > 0:
            tra.append(r["tra"])
        if r.get("latency_ms") is not None:
            lats.append(r["latency_ms"])
    return {
        "n": n,
        "exec_success": (1 - error_calls / total_calls) if total_calls else None,
        "total_calls": total_calls,
        "avg_turns": (sum(turns) / len(turns)) if turns else None,
        "avg_tokens": (sum(toks) / len(toks)) if toks else None,
        "em": em / n,
        "cost_per_correct": ((sum(toks) / len(toks)) / (em / n)) if em else None,
        "tra": (sum(tra) / len(tra)) if tra else None,
        "avg_latency_ms": (sum(lats) / len(lats)) if lats else None,
    }


def main():
    files = sorted(glob.glob(os.path.join(RESULTS, "doc_*.jsonl")))
    header = f"{'实验':<44}{'n':>4}{'ExecSR':>10}{'turns':>7}{'tok':>7}{'EM':>6}{'tok/正解':>9}{'TRA':>6}{'秒/题':>7}"
    print(header)
    print("-" * len(header))
    for f in files:
        name = os.path.basename(f).replace("doc_", "").replace(".jsonl", "")[:42]
        m = extra_metrics(load(f))
        if not m:
            continue
        cp = f"{m['exec_success']*100:.0f}%" if m["exec_success"] is not None else "N/A"
        tu = f"{m['avg_turns']:.1f}" if m["avg_turns"] is not None else "N/A"
        to = f"{m['avg_tokens']:.0f}" if m["avg_tokens"] is not None else "N/A"
        cc = f"{m['cost_per_correct']:.0f}" if m["cost_per_correct"] is not None else "N/A"
        tr = f"{m['tra']*100:.0f}%" if m["tra"] is not None else "N/A"
        la = f"{m['avg_latency_ms']/1000:.1f}" if m["avg_latency_ms"] is not None else "N/A"
        print(f"{name:<44}{m['n']:>4}{cp:>10}{tu:>7}{to:>7}{m['em']*100:>5.0f}%{cc:>9}{tr:>6}{la:>7}")


if __name__ == "__main__":
    main()
