# -*- coding: utf-8 -*-
"""三档对比 + CPRP（v9 的主分析脚本）。

三档（按文件名前缀识别）：
  doc_*   档 1 文档（不训练）
  iso_*   档 2 隔离（每工具独立模型，分别评测）
  full_*  档 3 全量（共享 LoRA / 续训）

输出：
  ① 三档表：EM_old（旧题=遗忘）/ EM_new（新工具）/ Forget / 成本
  ② CPRP：修复 1pp 遗忘所需的重放数据量（或 GPU 时）

用法：
  python analyze_three_tiers.py --dir results
"""
import argparse
import glob
import json
import os
import re

# 每档的"训练成本"配置（手工填，单位：重放数据条数 / GPU 时）
TIER_CONFIG = {
    "doc":  {"label": "档1 文档", "replay": 0,    "gpu_h": 0.0},
    "iso":  {"label": "档2 隔离", "replay": 0,    "gpu_h": 0.06},   # 每工具一次 SFT
    "full": {"label": "档3 全量", "replay": 0,    "gpu_h": 0.06},   # 续训一次
}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def em(rows):
    if not rows:
        return None
    return sum(1 for r in rows if float(r.get("em", 0)) >= 1.0) / len(rows)


def avg(rows, key, default=0):
    vals = [r.get(key, default) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def classify(fname):
    base = os.path.basename(fname)
    if base.startswith("doc_"):
        return "doc"
    if base.startswith("iso_"):
        return "iso"
    if base.startswith("full_"):
        return "full"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    buckets = {"doc": [], "iso": [], "full": []}
    for f in files:
        tier = classify(f)
        if tier:
            buckets[tier].append(f)

    print(f"{'档':<10}{'文件数':>6}{'EM_old':>8}{'EM_new':>8}{'Forget':>8}{'tok/题':>8}{'秒/题':>8}")
    print("-" * 60)
    summary = {}
    for tier, fs in buckets.items():
        if not fs:
            continue
        old_rows, new_rows = [], []
        for f in fs:
            rows = load(f)
            if "T2" in f or "new" in f.lower() or "z3" in f.lower():
                new_rows += rows
            else:
                old_rows += rows
        e_old, e_new = em(old_rows), em(new_rows)
        forget = (1 - e_new / e_old) if (e_old and e_new is not None) else None
        tok = avg(old_rows + new_rows, "in_tok") or 0
        lat = avg(old_rows + new_rows, "latency_ms") or 0
        summary[tier] = {"em_old": e_old, "em_new": e_new, "forget": forget,
                         "tok": tok, "lat": lat / 1000, "files": len(fs)}
        lab = TIER_CONFIG[tier]["label"]
        f_str = f"{forget*100:.0f}%" if forget is not None else "N/A"
        print(f"{lab:<10}{len(fs):>6}{e_old*100 if e_old else 0:>7.0f}%{e_new*100 if e_new else 0:>7.0f}%{f_str:>8}{tok:>8.0f}{lat/1000:>8.2f}")

    # CPRP（需要"无重放 vs 重放"两组 full 数据；用文件名含 replay 识别）
    replay_files = [f for f in files if "replay" in f.lower()]
    if replay_files:
        print("\n=== CPRP ===")
        for f in replay_files:
            m = re.search(r"replay(\d+)", f.lower())
            n_replay = int(m.group(1)) if m else 0
            rows = load(f)
            e = em(rows)
            print(f"{os.path.basename(f)}: EM={e*100:.0f}% 重放={n_replay}条")
        print("CPRP = 重放条数 / 修复的 pp（如 300 条修复 12pp → 25 条/pp）")
    else:
        print("\n[提示] 未找到 replay 结果文件（CPRP 需要'无重放 vs 重放'两组数据）")


if __name__ == "__main__":
    main()
