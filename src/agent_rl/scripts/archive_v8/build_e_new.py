# -*- coding: utf-8 -*-
"""方案 B：构造 E_new = AIME 2024（30 题，官方答案）+ 手写 20 题（python 必要性）。

流程：
  ① 在线拉 AIME 验证集（AI-MO/aimo-validation-aime，90 题 2022-2024）→ 筛 2024_AIME（30 题）
  ② 手写 20 题（循环/聚合/数论——calc 难以一步完成）
  ③ gold 自检：手写题的答案用 python 现算核对（AIME 用官方答案）
  ④ 落盘 data/e_new_python_tasks.jsonl（评测时离线可用）

用法（需网络一次）：
  HF_ENDPOINT=https://hf-mirror.com python build_e_new.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "e_new_python_tasks.jsonl")
AIME_CACHE = os.path.join(HERE, "data", "aime_2024_30.jsonl")

# 手写 20 题：(question, 期望答案, 用于自检的 python 表达式)
MANUAL = [
    ("What is the sum of the first 10 prime numbers?",
     129, "sum([p for p in range(2,100) if all(p%d for d in range(2,p))][:10])"),
    ("How many prime numbers are there between 1 and 100?",
     25, "len([p for p in range(2,101) if all(p%d for d in range(2,p))])"),
    ("How many divisors does 360 have?",
     24, "len([d for d in range(1,361) if 360%d==0])"),
    ("What is the greatest common divisor of 84 and 126?",
     42, "import math; math.gcd(84,126)"),
    ("What is the least common multiple of 12, 18 and 30?",
     180, "import math; math.lcm(12,18,30)"),
    ("What is the sum of the squares of the first 10 positive integers?",
     385, "sum(i*i for i in range(1,11))"),
    ("What is the sum of the cubes of the first 5 positive integers?",
     225, "sum(i**3 for i in range(1,6))"),
    ("What is the 20th Fibonacci number, counting 1, 1, 2, 3, ... as the start?",
     6765, "(lambda f: f(f,20))(lambda s,n: 1 if n<=2 else s(s,n-1)+s(s,n-2))"),
    ("What is the product of all integers from 1 to 10?", 3628800, "import math; math.factorial(10)"),
    ("What is the sum of the digits of 2 raised to the power 20?", 31, "sum(int(c) for c in str(2**20))"),
    ("How many trailing zeros are there in 100 factorial?", 24,
     "sum(100//(5**k) for k in range(1,4) if 5**k<=100)"),
    ("What is the remainder when 2 raised to the power 100 is divided by 7?", 2, "(2**100)%7"),
    ("What is the 100th triangular number?", 5050, "100*101//2"),
    ("How many ways are there to choose 3 items from 10 distinct items?", 120,
     "import math; math.comb(10,3)"),
    ("How many 4-digit numbers can be formed using digits 1,2,3,4 without repetition?", 24,
     "import math; math.perm(4,4)"),
    ("How many subsets does a set of 8 elements have?", 256, "2**8"),
    ("How many diagonals does a convex 12-gon have?", 54, "12*(12-3)//2"),
    ("Compute 1234 times 5678.", 7006652, "1234*5678"),
    ("What is 2 to the power 30?", 1073741824, "2**30"),
    ("What is the sum of the digits of all numbers from 1 to 100?", 901,
     "sum(sum(int(c) for c in str(n)) for n in range(1,101))"),
]


import math


def _check(expr):
    """预导入 math 的 eval 检查（表达式内不得出现 import）。"""
    return eval(expr.replace("import math; ", ""), {"math": math})


def build_manual():
    """手写 20 题 + 自检 gold。"""
    rows = []
    for q, expect, pyexpr in MANUAL:
        actual = _check(pyexpr)
        status = "OK" if actual == expect else f"MISMATCH expect={expect} actual={actual}"
        print(f"[{status}] {q[:60]}")
        rows.append({"question": q, "gold_answer": f"\\boxed{{{expect}}}", "gold_value": expect,
                     "kind": "python_manual"})
    return rows


def load_aime_2024():
    """取 AIME 2024 的 30 题（官方答案）；优先本地缓存。"""
    if os.path.exists(AIME_CACHE):
        rows = [json.loads(l) for l in open(AIME_CACHE) if l.strip()]
        print(f"AIME 2024（缓存）: {len(rows)} 题")
        return rows
    from datasets import load_dataset
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    rows = []
    for x in ds:
        if "2024_AIME" in x["url"]:
            ans = str(x["answer"]).strip()
            rows.append({"question": x["problem"], "gold_answer": f"\\boxed{{{ans}}}",
                         "gold_value": ans, "kind": "aime_2024", "url": x["url"]})
    print(f"AIME 2024: {len(rows)} 题")
    with open(AIME_CACHE, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def main():
    manual = build_manual()
    aime = load_aime_2024()
    allrows = aime + manual
    with open(OUT, "w") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"E_new: {len(allrows)} 题（AIME2024 {len(aime)} + 手写 {len(manual)}）→ {OUT}")


if __name__ == "__main__":
    main()
