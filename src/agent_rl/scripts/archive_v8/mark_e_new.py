# -*- coding: utf-8 -*-
"""给 E_new_T2 的 50 题加 category 标记（python_required / calc_solvable / hard_reasoning）。

判定规则（严格）：
  python_required：calc 单步/式能力**无法**完成——需要循环/函数（素数、约数、gcd/lcm、阶乘、
                   组合/排列、数位和、尾零、序列）——safe_calc 无这些能力
  calc_solvable：纯算术/幂/简单和——`calc(expr)` 一次即可（如 1234*5678、2**30）
  hard_reasoning：需要多步推理的竞赛题（AIME）——calc 分多步**能**做，但难度在推理不在工具

用法：在服务器上跑，原地更新 data/e_new_python_tasks.jsonl（加 category 字段）
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "data", "e_new_python_tasks.jsonl")

# 手写 20 题的判定（按题面关键词）
PY_REQUIRED_KW = [
    "prime", "primes", "divisor", "gcd", "least common multiple", "Fibonacci",
    "factorial", "fibonacci", "sum of the digits", "trailing zeros", "choose",
    "without repetition", "subsets", "number of primes", "greatest common divisor",
]


def classify(q: str, kind: str) -> str:
    if kind == "aime_2024":
        return "hard_reasoning"
    ql = q.lower()
    for kw in PY_REQUIRED_KW:
        if kw in ql:
            return "python_required"
    # 兜底：手写题里剩下的是纯算术/幂 → calc 可解
    return "calc_solvable"


def main():
    rows = [json.loads(l) for l in open(PATH) if l.strip()]
    from collections import Counter
    cnt = Counter()
    for r in rows:
        cat = classify(r["question"], r.get("kind", ""))
        r["category"] = cat
        cnt[cat] += 1
    with open(PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"标记完成：{dict(cnt)}")
    print("\n各子集抽样：")
    for cat in ["python_required", "calc_solvable", "hard_reasoning"]:
        ex = [r for r in rows if r["category"] == cat][:3]
        print(f"[{cat}] {len([r for r in rows if r['category']==cat])} 题")
        for r in ex:
            print(f"   {r['question'][:70]}")


if __name__ == "__main__":
    main()
