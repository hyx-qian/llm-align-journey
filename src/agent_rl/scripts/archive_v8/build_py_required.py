# -*- coding: utf-8 -*-
"""补充 18 道「calc 绝对做不了」的 python 必要题，合并到 E_new 的 python_required 子集（共 30 题）。

判定：每题都需要循环/函数/大数——safe_calc（单步表达式，无循环/无阶乘/无组合/无素数）无法完成。
自检：每题用 python 现算。
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

NEW_PY = [
    ("What is the sum of the first 20 prime numbers?", lambda: sum(_primes(20))),
    ("How many prime numbers are there between 1 and 200?", lambda: len([p for p in _primes(50) if p <= 200])),
    ("How many perfect squares are there between 1 and 1000?", lambda: math.isqrt(1000)),
    ("How many digits are there in 2 raised to the power 50?", lambda: len(str(2 ** 50))),
    ("What is the sum of all odd numbers from 1 to 99?", lambda: sum(range(1, 100, 2))),
    ("What is the sum of all integers from 1 to 50 that are divisible by 3 or 5, counting multiples of 15 only once?",
     lambda: sum(v for v in range(1, 51) if v % 3 == 0 or v % 5 == 0)),
    ("How many digits are there in 10 factorial?", lambda: len(str(math.factorial(10)))),
    ("What is the units digit of the product of all prime numbers less than 100?",
     lambda: int(str(math.prod([p for p in _primes(30) if p < 100]))[-1])),
    ("How many integers from 1 to 1000 do not contain the digit 7?",
     lambda: len([v for v in range(1, 1001) if "7" not in str(v)])),
    ("What is the sum of the first 30 triangular numbers?",
     lambda: sum(i * (i + 1) // 2 for i in range(1, 31))),
    ("What is the largest prime number less than 50?", lambda: max([p for p in _primes(20) if p < 50])),
    ("How many even integers are there from 1 to 100 inclusive?", lambda: len([v for v in range(1, 101) if v % 2 == 0])),
    ("How many divisors does 1000 have?", lambda: len([d for d in range(1, 1001) if 1000 % d == 0])),
    ("What is the greatest common divisor of 36 and 48?", lambda: math.gcd(36, 48)),
    ("What is the least common multiple of 12 and 18?", lambda: math.lcm(12, 18)),
    ("How many trailing zeros are there in 25 factorial?",
     lambda: sum(25 // (5 ** k) for k in range(1, 4) if 5 ** k <= 25)),
    ("What is the value of 20 choose 10?", lambda: math.comb(20, 10)),
    ("How many integers from 1 to 100 have digits that sum to 10?",
     lambda: len([v for v in range(1, 101) if sum(int(c) for c in str(v)) == 10])),
]


def _primes(n):
    out, c = [], 2
    while len(out) < n:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out


def main():
    os.makedirs(DATA, exist_ok=True)
    # 先加载原始 E_new，保留 python_required 12 题
    src = os.path.join(DATA, "e_new_python_tasks.jsonl")
    rows = [json.loads(l) for l in open(src) if l.strip()]
    py_req = [r for r in rows if r.get("category") == "python_required"]
    rest = [r for r in rows if r.get("category") != "python_required"]

    added = []
    for q, fn in NEW_PY:
        g = fn()
        added.append({"question": q, "gold_answer": f"\\boxed{{{g}}}", "gold_value": g,
                      "kind": "python_manual", "category": "python_required"})
    print(f"原有 python_required {len(py_req)} + 新增 {len(added)} = {len(py_req)+len(added)} 题")

    allrows = py_req + added + rest
    with open(src, "w") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 单独导出"必须 python"子集（供 E3 重跑）
    sub = py_req + added
    out2 = os.path.join(DATA, "e_new_T2_python_only_30.jsonl")
    with open(out2, "w") as f:
        for r in sub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"python_only 子集: {len(sub)} 题 → {out2}")
    for r in sub[:3]:
        print(f"  {r['question'][:65]} | gold={r['gold_value']}")


if __name__ == "__main__":
    main()
