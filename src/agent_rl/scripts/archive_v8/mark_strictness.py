# -*- coding: utf-8 -*-
"""严格分档：给 E_new_T2 与 E_mix_T2 的题加 strictness 字段。

三档定义（关键：calc 是"单步表达式"，无循环/无阶乘/无组合/无素数）：
  python_only   —— **多步 calc 也不可行**（步数 > 20 或需要函数）：素数计数、大数阶乘、组合/排列、
                   尾零、数位和、完全平方数计数、长序列聚合
  calc_multistep—— 多步 calc 可行但低效（步数 ≤ 20）：前 10 素数之和、平方和 1..10、小阶乘
  calc_onestep  —— 一步 calc 即可：大数乘法、幂、简单和

Discover 指标只算 `python_only`；`calc_multistep` 用于"效率"分析。
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def classify(question: str) -> str:
    q = question.lower()

    # 1) 明确 python_only 模式
    if re.search(r"how many (prime|divisor|perfect square|even integer|integer)", q):
        return "python_only"
    if "how many digits" in q:
        return "python_only"
    if "trailing zeros" in q:
        return "python_only"
    if "choose" in q or "without repetition" in q or "subsets" in q:
        return "python_only"
    if "sum of the digits" in q:
        return "python_only"
    if "do not contain" in q or "divisible by" in q or "with digits" in q:
        return "python_only"
    if "units digit" in q:
        return "python_only"
    if "triangular" in q:
        return "python_only"
    if "fibonacci" in q:
        return "python_only"
    if "least common multiple" in q or "greatest common divisor" in q:
        return "python_only"

    # 2) 前 N 个素数/平方和/立方和/阶乘 —— 看 N
    m = re.search(r"first (\d+) prime", q)
    if m:
        n = int(m.group(1))
        return "calc_multistep" if n <= 20 else "python_only"
    m = re.search(r"first (\d+) positive integers", q)
    if m:
        n = int(m.group(1))
        return "calc_multistep" if n <= 20 else "python_only"
    m = re.search(r"(\d+) factorial|product of all integers from 1 to (\d+)", q)
    if m:
        n = int(m.group(1) or m.group(2))
        return "calc_multistep" if n <= 10 else "python_only"
    if "sum of all odd numbers" in q or "sum of all even numbers" in q:
        return "calc_multistep"
    if "sum of cubes" in q:
        return "calc_multistep"

    # 3) 一步/简单
    if re.search(r"compute \d+ times \d+|power \d+|squared|triangle", q):
        return "calc_onestep"
    if re.search(r"sum of all integers from 1 to \d+", q):
        return "calc_multistep"
    return "calc_onestep"


def mark(path, field="strictness"):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    from collections import Counter
    cnt = Counter()
    for r in rows:
        s = classify(r["question"])
        r[field] = s
        cnt[s] += 1
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{os.path.basename(path)}: {dict(cnt)}")
    # 导出 python_only 子集（E_new 用）
    return rows


def main():
    p1 = os.path.join(DATA, "e_new_python_tasks.jsonl")
    rows = mark(p1)
    py_only = [r for r in rows if r.get("strictness") == "python_only"]
    sub = os.path.join(DATA, "e_new_T2_python_only_strict.jsonl")
    with open(sub, "w") as f:
        for r in py_only:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"严格 python_only 子集: {len(py_only)} 题 → {sub}")

    p2 = os.path.join(DATA, "e_mix_T2_collab.jsonl")
    mark(p2)

    print("\n=== python_only 子集抽样 ===")
    for r in py_only[:5]:
        print(f"  {r['question'][:70]}")


if __name__ == "__main__":
    main()
