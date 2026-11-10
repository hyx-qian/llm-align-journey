# -*- coding: utf-8 -*-
"""sympy（符号运算）工具：方程求解/化简/展开/因式分解/求导/积分。

安全：用 sympy 的 parse_expr + 受限 local_dict（只允许单个符号变量），
     并限制表达式长度/字符（防注入巨大表达式）。
返回格式与 safe_calc 一致：成功返回结果字符串，失败返回 "ERROR: ..."
"""
import re

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

ALLOWED_OPS = {"solve", "simplify", "expand", "factor", "diff", "integrate"}
MAX_LEN = 200
SAFE_CHARS = re.compile(r"^[0-9a-zA-Z_+\-*/^().,\s=]+$")

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def safe_sympy(expr: str, op: str = "solve", var: str = "x") -> str:
    expr = (expr or "").strip()
    op = (op or "solve").strip().lower()
    var = (var or "x").strip() or "x"

    if not expr:
        return "ERROR: empty expression"
    if len(expr) > MAX_LEN:
        return f"ERROR: expression too long (>{MAX_LEN} chars)"
    if not SAFE_CHARS.match(expr):
        return "ERROR: expression contains unsupported characters"
    if op not in ALLOWED_OPS:
        return f"ERROR: unknown operation '{op}' (allowed: {sorted(ALLOWED_OPS)})"
    if not re.match(r"^[a-zA-Z_]\w{0,10}$", var):
        return "ERROR: invalid variable name"

    try:
        sym = sympy.Symbol(var)
        local = {var: sym}

        # 处理 "lhs = rhs"（solve 场景）：转为 lhs - rhs
        if op == "solve":
            if "=" in expr:
                lhs, rhs = expr.split("=", 1)
                e = parse_expr(lhs, local_dict=local, transformations=_TRANSFORMS) - \
                    parse_expr(rhs, local_dict=local, transformations=_TRANSFORMS)
            else:
                e = parse_expr(expr, local_dict=local, transformations=_TRANSFORMS)
            res = sympy.solve(e, sym)
        else:
            body = expr.replace("=", "-(") + ")" * expr.count("=") if "=" in expr else expr
            e = parse_expr(body, local_dict=local, transformations=_TRANSFORMS)
            res = {
                "simplify": lambda: sympy.simplify(e),
                "expand": lambda: sympy.expand(e),
                "factor": lambda: sympy.factor(e),
                "diff": lambda: sympy.diff(e, sym),
                "integrate": lambda: sympy.integrate(e, sym),
            }[op]()

        return str(res)[:500]
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    cases = [
        ('3*x + 5 = 20', 'solve', 'x', '[5]'),
        ('x**2 - 4', 'solve', 'x', '[-2, 2]'),
        ('x**2 + 2*x + 1', 'factor', 'x', '(x + 1)**2'),
        ('(x+1)**2', 'expand', 'x', 'x**2 + 2*x + 1'),
        ('3*x + 5', 'diff', 'x', '3'),
        ('x**2', 'integrate', 'x', 'x**3/3'),
        ('__import__("os")', 'solve', 'x', 'ERROR'),
        ('a' * 300, 'solve', 'x', 'ERROR'),
        ('3*x + 5 = 20', 'bogus_op', 'x', 'ERROR'),
    ]
    for expr, op, var, expect in cases:
        r = safe_sympy(expr, op, var)
        ok = ("ERROR" in r) if expect == "ERROR" else (expect.replace(" ", "") in r.replace(" ", ""))
        print(f"[{'OK' if ok else 'FAIL'}] {expr[:30]!r}/{op} -> {r[:70]!r}")
