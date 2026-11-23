# -*- coding: utf-8 -*-
"""第三阶段评测集 + E_old_T1 落地（含每步自检）。

产出：
  1. data/e_old_T1_200.jsonl     —— E_old_T1（GSM8K 尾部 200，calc 可解）
  2. data/e_old_T3_250.jsonl     —— E_old_T3 = E_old_T1(200) + E_new_T2(50) 拼接
  3. data/e_mix_T3_3tool.jsonl   —— E_mix_T3（三工具协作 30 题）
自检：
  - E_old_T1：题面与 gold 来自现成数据（GSM8K 官方），只查非空与数量
  - E_mix_T3：每题用 python+sympy 现算（两步参考解法）
"""
import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
x = sp.Symbol("x")


def primes(n):
    out, c = [], 2
    while len(out) < n:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out


# ---------------- E_mix_T3：三工具协作（calc / python / sympy 各至少两种） ----------------
# 模式：calc→sympy、sympy→calc、python→sympy、sympy→python
E_MIX_T3 = [
    # calc → sympy
    ("First compute 3*4 with calc to get N, then solve the equation N*x + 5 = 17. What is x?", lambda: sp.solve(3 * 4 * x + 5 - 17, x)[0]),
    ("First compute 20-8 with calc to get N, then solve N*x = 24. What is x?", lambda: sp.solve((20 - 8) * x - 24, x)[0]),
    ("First compute 5*6 with calc to get N, then solve x**2 = N and give the positive root.", lambda: max(sp.solve(x ** 2 - 5 * 6, x))),
    ("First compute 100/4 with calc to get N, then solve x + N = 40. What is x?", lambda: sp.solve(x + 100 / 4 - 40, x)[0]),
    ("First compute 9+7 with calc to get N, then solve 2*x - N = 0. What is x?", lambda: sp.solve(2 * x - (9 + 7), x)[0]),
    # sympy → calc
    ("Solve the equation 3*x - 7 = 14 to get x, then add 5 to x with calc. What is the result?", lambda: sp.solve(3 * x - 7 - 14, x)[0] + 5),
    ("Solve the equation x**2 - 25 = 0 and take the positive root, then multiply it by 3 with calc. What is the result?", lambda: max(sp.solve(x ** 2 - 25, x)) * 3),
    ("Differentiate x**2 + x and evaluate at x = 3, then subtract 4 with calc. What is the result?", lambda: sp.diff(x ** 2 + x, x).subs(x, 3) - 4),
    ("Integrate 2*x from 0 to 3, then add 10 with calc. What is the result?", lambda: sp.integrate(2 * x, (x, 0, 3)) + 10),
    ("Simplify (x+2)**2 at x = 1, then divide by 3 with calc. What is the result?", lambda: sp.expand((x + 2) ** 2).subs(x, 1) / 3),
    # python → sympy
    ("First find the sum of the first 10 prime numbers with code to get N, then solve x - N = 0. What is x?", lambda: sp.solve(x - sum(primes(10)), x)[0]),
    ("First find 5 factorial with code to get N, then solve x/2 = N. What is x?", lambda: sp.solve(x / 2 - 120, x)[0]),
    ("First count the primes below 30 with code to get N, then solve x**2 = N and give the positive root.", lambda: max(sp.solve(x ** 2 - len([p for p in primes(15) if p < 30]), x))),
    ("First find the sum of integers 1..20 with code to get N, then solve x + 100 = N. What is x?", lambda: sp.solve(x + 100 - sum(range(1, 21)), x)[0]),
    ("First find 2**10 with code to get N, then solve 4*x = N. What is x?", lambda: sp.solve(4 * x - 1024, x)[0]),
    ("First find the 10th Fibonacci number with code to get N, then solve x - N = 0. What is x?", lambda: sp.solve(x - (lambda f: f(f, 10))(lambda s, n: 1 if n <= 2 else s(s, n - 1) + s(s, n - 2)), x)[0]),
    ("First find the sum of squares 1..5 with code to get N, then solve x**2 = N and give the positive root.", lambda: max(sp.solve(x ** 2 - sum(i * i for i in range(1, 6)), x))),
    ("First find 3**4 with code to get N, then solve x - N/3 = 0. What is x?", lambda: sp.solve(x - 3 ** 4 / 3, x)[0]),
    # sympy → python
    ("Solve x**2 - 9 = 0 and take the positive root N, then find N! with code. What is the factorial?",
     lambda: __import__("math").factorial(int(max(sp.solve(x ** 2 - 9, x))))),
    ("Solve 2*x + 1 = 11 to get x, then find the sum of the first x prime numbers with code. What is the sum?",
     lambda: sum(primes(int(sp.solve(2 * x + 1 - 11, x)[0])))),
    ("Differentiate x**3 and evaluate at x = 2 to get N, then find 2**N with code. What is the value?",
     lambda: 2 ** int(sp.diff(x ** 3, x).subs(x, 2))),
    ("Integrate x from 0 to 4 to get N, then find N! with code. What is the factorial?",
     lambda: __import__("math").factorial(int(sp.integrate(x, (x, 0, 4))))),
    ("Simplify (x+1)**2 at x = 2 to get N, then find the sum of digits of N**2 with code. What is it?",
     lambda: sum(int(c) for c in str(int(sp.expand((x + 1) ** 2).subs(x, 2)) ** 2))),
    ("Solve 5*x = 45 to get x, then find the sum of integers 1..x with code. What is the sum?",
     lambda: sum(range(1, int(sp.solve(5 * x - 45, x)[0]) + 1))),
    ("Factor x**2 - 4 and evaluate at x = 3 to get N, then find N**3 with code. What is the value?",
     lambda: int(sp.factor(x ** 2 - 4).subs(x, 3)) ** 3),
    ("Solve x + 7 = 19 to get x, then find the number of divisors of x with code. How many?",
     lambda: len([d for d in range(1, int(sp.solve(x + 7 - 19, x)[0]) + 1) if int(sp.solve(x + 7 - 19, x)[0]) % d == 0])),
    ("Integrate 3*x**2 from 0 to 2 to get N, then find the sum of digits of N*10 with code. What is it?",
     lambda: sum(int(c) for c in str(int(sp.integrate(3 * x ** 2, (x, 0, 2))) * 10))),
    ("Solve 4*x - 8 = 0 to get x, then compute x**5 with code. What is the value?",
     lambda: int(sp.solve(4 * x - 8, x)[0]) ** 5),
    ("Simplify (x-1)**2 at x = 4 to get N, then find the number of primes below N with code. How many?",
     lambda: len([p for p in primes(20) if p < int(sp.expand((x - 1) ** 2).subs(x, 4))])),
    ("Solve x/5 = 3 to get x, then find the sum of squares 1..x with code. What is the sum?",
     lambda: sum(i * i for i in range(1, int(sp.solve(x / 5 - 3, x)[0]) + 1))),
    ("Differentiate x**4 at x = 1 to get N, then find the sum of digits of N**2 with code. What is it?",
     lambda: sum(int(c) for c in str(int(sp.diff(x ** 4, x).subs(x, 1)) ** 2))),
    ("Solve 3*x + 6 = 21 to get x, then find the sum of all integers from 1 to x**2 with code. What is the sum?",
     lambda: sum(range(1, int(sp.solve(3 * x + 6 - 21, x)[0]) ** 2 + 1))),
    ("Integrate 2*x from 0 to 5 to get N, then find the sum of digits of N**2 with code. What is it?",
     lambda: sum(int(c) for c in str(int(sp.integrate(2 * x, (x, 0, 5))) ** 2))),
]


def build_old_t1(n=200):
    """E_old_T1：GSM8K agent_train_4k 尾部 200 条（calc 可解）。"""
    src = os.path.join(DATA, "gsm8k_agent_train_4k.jsonl")
    rows = [json.loads(l) for l in open(src) if l.strip()][-n:]
    out = []
    for r in rows:
        q = next((m["content"] for m in r["prompt"] if m["role"] == "user"), "")
        gold = r.get("gold_answer", "")
        assert q and gold, "空题或空 gold"
        out.append({"question": q, "gold_answer": gold, "source": "gsm8k_tail"})
    path = os.path.join(DATA, "e_old_T1_200.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"E_old_T1: {len(out)} 题 → {path}")
    return out


def build_old_t3(old_t1):
    """E_old_T3 = E_old_T1(200) + E_new_T2(50)。"""
    new_t2_path = os.path.join(DATA, "e_new_python_tasks.jsonl")
    new_t2 = [json.loads(l) for l in open(new_t2_path) if l.strip()]
    out = [{"question": r["question"], "gold_answer": r["gold_answer"], "source": "old_T1"} for r in old_t1]
    out += [{"question": r["question"], "gold_answer": r["gold_answer"], "source": "old_T2_python"} for r in new_t2]
    path = os.path.join(DATA, "e_old_T3_250.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"E_old_T3: {len(out)} 题（{len(old_t1)} + {len(new_t2)}）→ {path}")


def build_mix_t3():
    os.makedirs(DATA, exist_ok=True)
    out, bad = [], 0
    for i, (q, fn) in enumerate(E_MIX_T3):
        try:
            g = fn()
            if isinstance(g, sp.Basic):
                g = sp.simplify(g)
            gf = float(g)
            g = int(gf) if gf == int(gf) else round(gf, 6)
        except Exception as e:
            print(f"[FAIL-T3mix-{i}] {q[:55]} -> {type(e).__name__}: {e}")
            bad += 1
            continue
        out.append({"question": q, "gold_answer": f"\\boxed{{{g}}}", "gold_value": g, "source": "e_mix_T3"})
    path = os.path.join(DATA, "e_mix_T3_3tool.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"E_mix_T3: {len(out)} 题（bad={bad}）→ {path}")


def main():
    os.makedirs(DATA, exist_ok=True)
    old_t1 = build_old_t1(200)
    build_old_t3(old_t1)
    build_mix_t3()
    print("\n=== 抽样 ===")
    for fname in ["e_old_T1_200.jsonl", "e_old_T3_250.jsonl", "e_mix_T3_3tool.jsonl"]:
        r = json.loads(open(os.path.join(DATA, fname)).readline())
        print(f"[{fname}] {r['question'][:60]} | gold={r.get('gold_answer', '')[:30]}")


if __name__ == "__main__":
    main()
