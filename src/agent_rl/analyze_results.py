# -*- coding: utf-8 -*-
"""结果分析与成本核算：读各实验 JSONL → 单实验指标 + 演化指标 + 成本表。

用法：
  python analyze_results.py                     # 自动扫描 results/*.jsonl
  python analyze_results.py --pair T1=E1_xxx.jsonl T2=E4_yyy.jsonl   # 指定配对算演化指标

指标定义：
  单实验：format_rate / exec_rate / EM / ans=None / stop_reason / tokens
  演化：Forget = 1 - EM_old(T2)/EM_old(T1)
        Discover = EM_new(T2)
        BWT = EM_old(T2) - EM_old(T1)
        FWT = EM_new(T2) - EM_new(T1)
        AvgPerf = mean(EM_old, EM_new, EM_mix)
  成本：训练 GPU 时（手工传入）+ 推理 tokens（自动统计）→ 相对成本（以基准为 1.0）
"""
import argparse
import glob
import json
import os
import statistics
from collections import Counter


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def basic_metrics(rows):
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "format_rate": sum(1 for r in rows if r.get("format")) / n,
        "exec_rate": sum(1 for r in rows if r.get("exec")) / n,
        "EM": sum(1 for r in rows if float(r.get("em", 0)) >= 1.0) / n,
        "ans_none": sum(1 for r in rows if r.get("ans") is None) / n,
        "stop_reason": dict(Counter(r.get("stop_reason", "?") for r in rows)),
        "in_tokens": statistics.mean([r.get("in_tok", 0) for r in rows]) if rows else 0,
        "out_tokens": statistics.mean([r.get("out_tok", 0) for r in rows]) if rows else 0,
    }


def print_exp_table(items):
    print("\n=== 单实验指标 ===")
    hdr = f"{'实验':<26}{'n':>5}{'format':>8}{'exec':>8}{'EM':>8}{'none':>7}{'in_tok':>8}{'out_tok':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in items:
        if not m:
            continue
        print(f"{name:<26}{m['n']:>5}{m['format_rate']*100:>7.0f}%{m['exec_rate']*100:>7.0f}%"
              f"{m['EM']*100:>7.0f}%{m['ans_none']*100:>6.0f}%{m['in_tokens']:>8.0f}{m['out_tokens']:>8.0f}")


def evolve_metrics(t1_old, t2_old, t1_new, t2_new, mix):
    """算演化指标（输入为 rows 列表；缺失则跳过）。"""
    em = lambda rows: (sum(1 for r in rows if float(r.get("em", 0)) >= 1.0) / len(rows)) if rows else None
    e1_old, e2_old = em(t1_old), em(t2_old)
    e1_new, e2_new = em(t1_new), em(t2_new)
    e_mix = em(mix)
    out = {}
    if e1_old and e2_old is not None:
        out["Forget"] = 1 - e2_old / e1_old if e1_old else None
        out["BWT"] = e2_old - e1_old
    if e2_new is not None:
        out["Discover"] = e2_new
    if e1_new is not None and e2_new is not None:
        out["FWT"] = e2_new - e1_new
    perf = [v for v in [e2_old, e2_new, e_mix] if v is not None]
    if perf:
        out["AvgPerf"] = sum(perf) / len(perf)
    return {k: round(v, 4) for k, v in out.items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--pair", nargs="*", help="如 T1=E1.jsonl T2=E4.jsonl")
    ap.add_argument("--train-hours", nargs="*", default=[], help="实验名=GPU小时（如 E6=0.05）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    items = []
    for f in files:
        name = os.path.basename(f).replace(".jsonl", "")
        if name.startswith("verify_") or name.startswith("smoke_"):
            continue
        items.append((name, basic_metrics(load(f))))
    print_exp_table(items)

    if args.pair:
        mapping = {}
        for kv in args.pair:
            k, v = kv.split("=", 1)
            mapping[k] = load(v)
        if all(k in mapping for k in ["T1", "T2"]):
            print("\n=== 演化指标（T1 vs T2）===")
            m = evolve_metrics(mapping.get("T1"), mapping.get("T2"),
                               mapping.get("T1_new"), mapping.get("T2_new"), mapping.get("MIX"))
            for k, v in m.items():
                print(f"{k}: {v*100:.1f}%")

    if args.train_hours:
        print("\n=== 成本（相对）===")
        th = dict(x.split("=") for x in args.train_hours)
        base = None
        rows = []
        for name, hours in th.items():
            m = dict(items).get(name, {})
            infer = (m.get("in_tokens", 0) + m.get("out_tokens", 0))
            cost = float(hours) * 3600 + infer  # 训练秒 + 每题 token（简化相对成本）
            rows.append((name, float(hours), infer, cost))
        base = min(r[3] for r in rows) if rows else 1
        for name, h, inf, c in rows:
            print(f"{name}: 训练 {h:.3f} GPU时 | 推理 {inf:.0f} tok/题 | 相对成本 ×{c/base:.2f}")


if __name__ == "__main__":
    main()
