# -*- coding: utf-8 -*-
"""构造 E_mix_T2（calc+python 协作题 30）与 E_new_T3（sympy 必要题 30），每题独立自检。

定义（严格）：
  E_mix_T2：**单题需要 calc 与 python 各至少一次，且第二步依赖第一步结果**——"协作"≠"两个工具都能做"。
  E_new_T3：**只有 sympy 能做**（方程/化简求值/求导/积分）；python sandbox 白名单不含 sympy，故 python 也做不了。

自检方式（独立于 gold 的构造）：
  - E_mix_T2：用 python 按"两步参考解法"现算（calc 那步用普通算术替代）
  - E_new_T3：用 **sympy** 真正求解/化简/求导/积分（不是手写常数）
"""
import json
import math
import os

import sympy as sp

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
x = sp.Symbol("x")


def primes(n):
    out, c = [], 2
    while len(out) < n:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out


def digit_sum(n):
    return sum(int(c) for c in str(n))


# ---------------- E_mix_T2：calc + python 协作（30 题，两步依赖） ----------------
E_MIX = [
    ("First compute 48/2 with calc, then use that result N to find the sum of the first N prime numbers. What is the sum?", "calc→python", lambda: sum(primes(int(48 / 2)))),
    ("First compute 20+30 with calc, then use that result N to find the sum of the first N prime numbers. What is the sum?", "calc→python", lambda: sum(primes(int(20 + 30)))),
    ("First compute 12*3 with calc, then find the sum of squares of the first N positive integers (N = that product). What is the sum?", "calc→python", lambda: sum(i * i for i in range(1, 12 * 3 + 1))),
    ("First compute 100/4 with calc, then count how many prime numbers are less than that result N. How many?", "calc→python", lambda: len([p for p in primes(30) if p < 100 // 4])),
    ("First compute 9*4 with calc, then compute (N/2)! where N is that product. What is the factorial?", "calc→python", lambda: math.factorial(int(9 * 4 / 2))),
    ("First compute 15+25 with calc, then find the sum of all integers from 1 to N (N = that sum). What is the sum?", "calc→python", lambda: sum(range(1, 15 + 25 + 1))),
    ("First compute 144/12 with calc, then compute 2 to the power N (N = that result). What is the value?", "calc→python", lambda: 2 ** int(144 / 12)),
    ("First compute 6*7 with calc, then find the number of divisors of that product N. How many divisors?", "calc→python", lambda: len([d for d in range(1, 6 * 7 + 1) if (6 * 7) % d == 0])),
    # python → calc
    ("First find the sum of the first 10 prime numbers with code, then subtract 100 using a calculator. What is the result?", "python→calc", lambda: sum(primes(10)) - 100),
    ("First find the sum of all integers from 1 to 100 with code, then divide it by 2 using a calculator. What is the result?", "python→calc", lambda: int(sum(range(1, 101)) / 2)),
    ("First find 10 factorial with code, then divide it by 1000000 using a calculator. What is the result?", "python→calc", lambda: int(math.factorial(10) / 1000000)),
    ("First find the sum of the squares of 1..10 with code, then add 15 with a calculator. What is the result?", "python→calc", lambda: sum(i * i for i in range(1, 11)) + 15),
    ("First count how many primes are below 50 with code, then multiply that count by 7 using a calculator. What is the result?", "python→calc", lambda: len([p for p in primes(20) if p < 50]) * 7),
    ("First compute 3 to the power 10 with code, then subtract 50000 using a calculator. What is the result?", "python→calc", lambda: 3 ** 10 - 50000),
    ("First find the 15th Fibonacci number with code, then multiply it by 4 using a calculator. What is the result?", "python→calc", lambda: (lambda f: f(f, 15))(lambda s, n: 1 if n <= 2 else s(s, n - 1) + s(s, n - 2)) * 4),
    ("First find the sum of digits of 2^20 with code, then add 9 using a calculator. What is the result?", "python→calc", lambda: digit_sum(2 ** 20) + 9),
    # 两步同类工具交替（仍是两工具）
    ("Compute 25*4 with calc to get N, then find the sum of all even numbers from 2 to N with code. What is the sum?", "calc→python", lambda: sum(i for i in range(2, 25 * 4 + 1, 2))),
    ("Compute 50-14 with calc to get N, then compute N! with code. What is the factorial?", "calc→python", lambda: math.factorial(50 - 14)),
    ("Compute 81/9 with calc to get N, then find the N-th triangular number with code. What is it?", "calc→python", lambda: (81 // 9) * (81 // 9 + 1) // 2),
    ("Compute 11*11 with calc to get N, then count the perfect squares up to N with code. How many?", "calc→python", lambda: math.isqrt(11 * 11)),
    ("Compute 5*5*5 with calc to get N, then find the sum of the digits of N squared with code. What is it?", "calc→python", lambda: digit_sum((5 * 5 * 5) ** 2)),
    ("Compute 30+70 with calc to get N, then count integers from 1 to N divisible by 7 with code. How many?", "calc→python", lambda: len([v for v in range(1, 30 + 70 + 1) if v % 7 == 0])),
    ("Compute 200/8 with calc to get N, then find the sum of the first N odd numbers with code. What is the sum?", "calc→python", lambda: sum(2 * i + 1 for i in range(200 // 8))),
    ("Compute 13+17 with calc to get N, then compute N choose 5 with code. What is the value?", "calc→python", lambda: math.comb(13 + 17, 5)),
    ("Compute 64/2 with calc to get N, then find the largest prime factor of N with code. What is it?", "calc→python", lambda: max([p for p in primes(20) if (64 // 2) % p == 0])),
    ("Compute 8*8 with calc to get N, then find the sum of all divisors of N with code. What is the sum?", "calc→python", lambda: sum(d for d in range(1, 8 * 8 + 1) if (8 * 8) % d == 0)),
    ("Compute 19-4 with calc to get N, then compute the product of the first N positive integers with code. What is the product?", "calc→python", lambda: math.factorial(19 - 4)),
    ("Compute 3*14 with calc to get N, then find the sum of the squares of the digits of N with code. What is it?", "calc→python", lambda: sum(int(c) ** 2 for c in str(3 * 14))),
    ("Compute 7*9 with calc to get N, then find the number of trailing zeros of N! with code. How many?", "calc→python", lambda: sum((7 * 9) // (5 ** k) for k in range(1, 5) if 5 ** k <= 7 * 9)),
    ("Compute 1000/40 with calc to get N, then find the sum of all primes less than N with code. What is the sum?", "calc→python", lambda: sum(p for p in primes(30) if p < 1000 // 40)),
    ("Compute 16+19 with calc to get N, then find the sum of the digits of N cubed with code. What is it?", "calc→python", lambda: digit_sum((16 + 19) ** 3)),
]

# ---------------- E_new_T3：只有 sympy 能做（30 题，sympy 独立验证） ----------------
E_T3 = [
    ("Solve the equation 3*x + 5 = 20. What is x?", lambda: sp.solve(3 * x + 5 - 20, x)[0]),
    ("Solve the equation 2*x - 7 = 15. What is x?", lambda: sp.solve(2 * x - 7 - 15, x)[0]),
    ("Solve the equation 5*x + 3 = 28. What is x?", lambda: sp.solve(5 * x + 3 - 28, x)[0]),
    ("Solve the equation x/4 + 6 = 10. What is x?", lambda: sp.solve(x / 4 + 6 - 10, x)[0]),
    ("Solve the equation 7*x - 4 = 45. What is x?", lambda: sp.solve(7 * x - 4 - 45, x)[0]),
    ("Solve the equation x**2 - 16 = 0 and give the positive root. What is it?", lambda: max(sp.solve(x ** 2 - 16, x))),
    ("Solve the equation x**2 - 5*x + 6 = 0 and give the smaller root. What is it?", lambda: min(sp.solve(x ** 2 - 5 * x + 6, x))),
    ("Solve the equation x**2 + 7*x + 12 = 0 and give the larger root. What is it?", lambda: max(sp.solve(x ** 2 + 7 * x + 12, x))),
    ("Solve the equation 2*x**2 - 8 = 0 and give the positive root. What is it?", lambda: max(sp.solve(2 * x ** 2 - 8, x))),
    ("Solve the equation 4*x - 3 = 2*x + 9. What is x?", lambda: sp.solve(4 * x - 3 - (2 * x + 9), x)[0]),
    ("Simplify (x+1)**2 - x**2, then evaluate at x = 5. What is the value?", lambda: sp.simplify((x + 1) ** 2 - x ** 2).subs(x, 5)),
    ("Simplify (x+2)*(x-2) - x**2, then evaluate at x = 3. What is the value?", lambda: sp.simplify((x + 2) * (x - 2) - x ** 2).subs(x, 3)),
    ("Expand (x+3)**2, then evaluate at x = 2. What is the value?", lambda: sp.expand((x + 3) ** 2).subs(x, 2)),
    ("Simplify x**3/x for x != 0, then evaluate at x = 7. What is the value?", lambda: sp.simplify(x ** 3 / x).subs(x, 7)),
    ("Simplify (2*x+4)/2, then evaluate at x = 8. What is the value?", lambda: sp.simplify((2 * x + 4) / 2).subs(x, 8)),
    ("Expand (x-1)**3, then evaluate at x = 3. What is the value?", lambda: sp.expand((x - 1) ** 3).subs(x, 3)),
    ("Simplify (x**2-9)/(x-3) for x != 3, then evaluate at x = 5. What is the value?", lambda: sp.simplify((x ** 2 - 9) / (x - 3)).subs(x, 5)),
    ("Simplify (x+1)*(x-1), then evaluate at x = 10. What is the value?", lambda: sp.expand((x + 1) * (x - 1)).subs(x, 10)),
    ("Simplify (3*x**2 + 6*x)/3, then evaluate at x = 4. What is the value?", lambda: sp.simplify((3 * x ** 2 + 6 * x) / 3).subs(x, 4)),
    ("Expand (x+5)**2, then evaluate at x = 1. What is the value?", lambda: sp.expand((x + 5) ** 2).subs(x, 1)),
    ("Differentiate x**3, then evaluate the derivative at x = 2. What is the value?", lambda: sp.diff(x ** 3, x).subs(x, 2)),
    ("Differentiate x**2 + 3*x, then evaluate at x = 4. What is the value?", lambda: sp.diff(x ** 2 + 3 * x, x).subs(x, 4)),
    ("Differentiate 5*x**3 - 2*x, then evaluate at x = 1. What is the value?", lambda: sp.diff(5 * x ** 3 - 2 * x, x).subs(x, 1)),
    ("Differentiate x**4, then evaluate at x = 2. What is the value?", lambda: sp.diff(x ** 4, x).subs(x, 2)),
    ("Differentiate 3*x**2 - 4*x + 1, then evaluate at x = 5. What is the value?", lambda: sp.diff(3 * x ** 2 - 4 * x + 1, x).subs(x, 5)),
    ("Integrate x**2 from 0 to 3. What is the value?", lambda: sp.integrate(x ** 2, (x, 0, 3))),
    ("Integrate 2*x from 0 to 4. What is the value?", lambda: sp.integrate(2 * x, (x, 0, 4))),
    ("Integrate 3*x**2 from 1 to 2. What is the value?", lambda: sp.integrate(3 * x ** 2, (x, 1, 2))),
    ("Integrate x from 0 to 6. What is the value?", lambda: sp.integrate(x, (x, 0, 6))),
    ("Integrate 4*x**3 from 0 to 2. What is the value?", lambda: sp.integrate(4 * x ** 3, (x, 0, 2))),
]


def build(name, rows, outfile, has_note=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    out, bad = [], 0
    for i, row in enumerate(rows):
        q, fn = row[0], row[-1]
        note = row[1] if has_note and len(row) == 3 else ""
        try:
            gold = fn()
            if isinstance(gold, sp.Basic):
                gold = sp.simplify(gold)
            try:
                gf = float(gold)
                gold = int(gf) if gf == int(gf) else round(gf, 6)
            except (TypeError, ValueError):
                pass
        except Exception as e:
            print(f"[FAIL-{name}-{i}] {q[:50]} -> {type(e).__name__}: {e}")
            bad += 1
            continue
        out.append({"question": q, "gold_answer": f"\\boxed{{{gold}}}", "gold_value": gold, "note": note, "source": name})
    path = os.path.join(OUT_DIR, outfile)
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(out)} 题（bad={bad}）→ {path}")
    return out


def main():
    mix = build("E_mix_T2", E_MIX, "e_mix_T2_collab.jsonl")
    t3 = build("E_new_T3", E_T3, "e_new_T3_sympy.jsonl", has_note=False)
    for tag, rows in [("E_mix_T2", mix), ("E_new_T3", t3)]:
        print(f"\n[{tag}] 抽样 3 题:")
        for r in rows[:3]:
            print(f"  {r['question'][:75]} | gold={r['gold_value']}")


if __name__ == "__main__":
    main()
