# -*- coding: utf-8 -*-
"""Z3 约束求解工具（v9 数学推理场景 T1/T2 的核心工具）。

【主任务】v9 双场景工具演化：
  数学场景 T1 = {sympy} → T2 = {sympy, z3}（新工具上线触发演化）
  Z3 = SMT 约束求解——"3x+2y=17 的正整数解个数"心算不可行，必须枚举/推理。

接口（与 safe_calc 一致，返回字符串）：
  safe_z3(constraints: str, var_spec: str = "x:Int") -> str
  · constraints = "3*x + 2*y == 17, x > 0, y > 0, x < 10"
  · var_spec    = "x,y:Int"（共享类型）或 "x:Int,y:Real"（逐个）
  · 返回："x=1, y=7" / "unsat" / "ERROR: ..."

安全：超时 5s + 约束≤200 字符 + 变量≤5 + 受限 eval（__builtins__ 清空）。
"""
import re

try:
    import z3
except ImportError:
    z3 = None

MAX_LEN = 200
MAX_VARS = 5
MAX_SOLUTIONS = 3
TIMEOUT_MS = 5000

_IDENT = re.compile(r"^[a-zA-Z_][\w]{0,15}$")


def _parse_var_spec(var_spec: str) -> dict:
    """解析 'x,y:Int' / 'x:Int,y:Real' → {"x": z3.Int("x"), ...}"""
    spec = (var_spec or "").strip()
    if not spec:
        raise ValueError("empty var_spec")
    env = {}

    # 形式 A："a,b,c:Int"（共享类型）
    m = re.match(r"^([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)\s*:\s*(Int|Real)$", spec)
    if m:
        names = [n.strip() for n in m.group(1).split(",")]
        typ = m.group(2)
        for n in names:
            if not _IDENT.match(n):
                raise ValueError(f"invalid var name '{n}'")
            env[n] = z3.Int(n) if typ == "Int" else z3.Real(n)
    else:
        # 形式 B："x:Int,y:Real"
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"missing type in '{part}'")
            name, typ = [s.strip() for s in part.split(":", 1)]
            if not _IDENT.match(name):
                raise ValueError(f"invalid var name '{name}'")
            if typ not in ("Int", "Real"):
                raise ValueError(f"unsupported type '{typ}'")
            env[name] = z3.Int(name) if typ == "Int" else z3.Real(name)

    if not env:
        raise ValueError("no variables parsed")
    if len(env) > MAX_VARS:
        raise ValueError(f"too many variables (>{MAX_VARS})")
    return env


def _build_expr(constraints: str, env: dict):
    """'3*x + 2*y == 17, x > 0' → [z3 约束, ...]（受限 eval，禁止任意代码）"""
    allowed = dict(env)
    allowed.update({"And": z3.And, "Or": z3.Or, "Not": z3.Not,
                    "Abs": z3.Abs, "If": z3.If, "True": True, "False": False})
    parts = [p.strip() for p in constraints.split(",") if p.strip()]
    if not parts:
        raise ValueError("no constraints")
    conds = []
    for p in parts:
        # Python 关键字 → z3
        p = re.sub(r"\band\b", "And", p)
        p = re.sub(r"\bor\b", "Or", p)
        p = re.sub(r"\bnot\b", "Not", p)
        # 禁止 dunder / import 等
        if re.search(r"__|\bimport\b|\blambda\b", p):
            raise ValueError("unsupported expression")
        try:
            v = eval(p, {"__builtins__": {}}, allowed)
        except Exception as e:
            raise ValueError(f"bad constraint '{p}': {type(e).__name__}")
        conds.append(v)
    return conds


def _format_model(model, env: dict) -> str:
    """model → 'x=1, y=7'"""
    parts = []
    for name, var in env.items():
        val = model[var] if model is not None else None
        parts.append(f"{name}={val if val is not None else 'any'}")
    return ", ".join(parts)


def safe_z3(constraints: str, var_spec: str = "x:Int") -> str:
    """对外接口：求解约束，返回解字符串（唯一解）或 'unsat' 或 'ERROR: ...'。"""
    if z3 is None:
        return "ERROR: z3-solver not installed (pip install z3-solver)"
    if not constraints or not constraints.strip():
        return "ERROR: empty constraints"
    if len(constraints) > MAX_LEN:
        return f"ERROR: constraints too long (>{MAX_LEN} chars)"
    try:
        env = _parse_var_spec(var_spec)
        solver = z3.Solver()
        solver.set("timeout", TIMEOUT_MS)
        conds = _build_expr(constraints, env)
        solver.add(*conds)

        result = solver.check()
        if result == z3.unsat:
            return "unsat"
        if result == z3.unknown:
            return "ERROR: solver unknown/timeout"

        # 枚举前 MAX_SOLUTIONS 个解（唯一解时只有一个）
        sols = []
        for _ in range(MAX_SOLUTIONS):
            if solver.check() != z3.sat:
                break
            m = solver.model()
            sols.append(_format_model(m, env))
            # 阻断当前解，找下一个
            solver.add(z3.Or(*[v != m[v] for v in env.values()]))

        if len(sols) == 1:
            return sols[0]
        return f"count>={len(sols)}: " + " | ".join(sols)
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    cases = [
        # (constraints, var_spec, 期望片段)
        ("3*x + 2*y == 17, x > 0, y > 0, x < 10", "x,y:Int", "x="),      # 多解（枚举）
        ("x + y == 10, x - y == 2", "x,y:Int", "x=6"),                   # 唯一解
        ("x*x == 4", "x:Int", "x="),                                     # 多解（2/-2）
        ("x > 5, x < 3", "x:Int", "unsat"),                              # 无解
        ("x + ", "x:Int", "ERROR"),                                      # 语法错
        ("x == 1", "x:Int", "x=1"),                                      # 最简单
    ]
    for c, v, expect in cases:
        r = safe_z3(c, v)
        ok = expect in r
        print(f"[{'OK' if ok else 'FAIL'}] {c[:42]:<42} | {v:<10} -> {r[:70]}")
