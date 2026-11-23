# -*- coding: utf-8 -*-
"""Z3 约束求解工具（v9 数学推理场景 T2 的新工具）。

【主任务上下文】
v9 研究：工具集演化时，1.5B + 参数级训练（+重放）能否成本更低、能力逼近 7B？
本工具位置：数学场景 T2 = {sympy} → {sympy, z3}——"新工具上线"触发演化。
为什么是 Z3：约束满足（如 "3x+2y=17 的正整数解"）心算不可行、必须枚举/推理——
             它是"不可被心算替代"的工具，才能测出遗忘与参数级训练收益。

接口（与 safe_calc 一致，返回字符串）：
  safe_z3(constraints: str, var_spec: str = "x:Int") -> str
  · constraints = "3*x + 2*y == 17, x > 0, y > 0, x < 10"
  · var_spec    = "x,y:Int"
  · 返回："x=1, y=7"（唯一解）/ "unsat" / "ERROR: ..."

安全（四条，必须）：
  ① 超时 5s（solver.set timeout）
  ② 约束 ≤200 字符
  ③ 变量 ≤5 个
  ④ 只允许 Int/Real + 算术/比较——**禁止 eval 任意代码**
"""
import re

try:
    import z3
except ImportError:  # 未安装时给出明确提示
    z3 = None

MAX_LEN = 200
MAX_VARS = 5
TIMEOUT_MS = 5000


# ============ TODO 1（学员）：解析变量规格 ============
def _parse_var_spec(var_spec: str) -> dict:
    """解析 'x,y:Int' 或 'x:Int,y:Real' → {"x": z3.Int("x"), "y": z3.Real("y")}

    TODO(学员) 实现要求：
      · 按 ',' 分段，每段形如 "x:Int" 或 "x"（缺省 Int）
      · 只允许 Int / Real 两种类型
      · 变量名只允许 [a-zA-Z_][a-zA-Z0-9_]*
      · 变量数 > MAX_VARS → raise ValueError
      · 兼容 "x,y:Int"（共享类型）
    """
    raise NotImplementedError("TODO 1: 实现变量规格解析")


# ============ TODO 2（学员）：构造 z3 约束 ============
def _build_expr(constraints: str, env: dict):
    """把 "3*x + 2*y == 17, x > 0" 转成 z3 约束（返回 z3 BoolRef 或 list）。

    TODO(学员) 实现要求：
      · 按 ',' 拆分多个约束（注意 == 里没有逗号，安全；但函数调用如 And(a,b) 有逗号——
        若支持 and/or 关键字可先只做单约束列表，不解析函数调用）
      · 把变量名替换为 env 里的 z3 变量后求值——**用受限 eval**：
          allowed = {**env, "And": z3.And, "Or": z3.Or, "Not": z3.Not,
                     "Abs": z3.Abs, "If": z3.If}
          return eval(expr_str, {"__builtins__": {}}, allowed)
        （__builtins__ 置空 = 禁止任意代码；字符集已由 safe_z3 的长度/变量数限制保护）
      · 任何异常 → raise ValueError（由 safe_z3 兜底成 ERROR 字符串）
    """
    raise NotImplementedError("TODO 2: 实现约束构造（受限 eval）")


# ============ TODO 3（学员）：格式化解 ============
def _format_model(model, env: dict) -> str:
    """把 z3 model 格式化成 "x=1, y=7"（多解时枚举前 3 个）。

    TODO(学员) 实现要求：
      · 遍历 env 的变量，model[v] 取值（None 表示该变量自由——可报 "x=any"）
      · 唯一解 → "x=1, y=7"
      · 多解 → 用 solver 重新 check + add(block) 枚举，最多 3 个，用 " | " 连接
        （若实现多解枚举较麻烦，先只报第一个解 + " (first of many)"，标注 TODO）
    """
    raise NotImplementedError("TODO 3: 实现模型格式化")


def safe_z3(constraints: str, var_spec: str = "x:Int") -> str:
    """对外接口（骨架已完成；核心逻辑在 TODO 1/2/3）。"""
    if z3 is None:
        return "ERROR: z3-solver not installed (pip install z3-solver)"
    if not constraints or not constraints.strip():
        return "ERROR: empty constraints"
    if len(constraints) > MAX_LEN:
        return f"ERROR: constraints too long (>{MAX_LEN} chars)"
    try:
        env = _parse_var_spec(var_spec)       # TODO 1
        solver = z3.Solver()
        solver.set("timeout", TIMEOUT_MS)
        cond = _build_expr(constraints, env)  # TODO 2
        solver.add(cond) if not isinstance(cond, list) else solver.add(*cond)

        result = solver.check()
        if result == z3.unsat:
            return "unsat"
        if result == z3.unknown:
            return "ERROR: solver unknown/timeout"
        return _format_model(solver.model(), env)  # TODO 3
    except NotImplementedError:
        raise
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    # 单测骨架（学员补：唯一解 / 多解 / unsat / 语法错 / 超长）
    cases = [
        ("3*x + 2*y == 17, x > 0, y > 0, x < 10", "x,y:Int", "x="),
        # TODO(学员) 补 4 个用例：
        #   ("x + y == 10, x - y == 2", "x,y:Int", "x=6")   # 唯一解
        #   ("x*x == 4", "x:Int", "x=2")                      # 多解（2/-2）
        #   ("x > 5, x < 3", "x:Int", "unsat")                # 无解
        #   ("x + ", "x:Int", "ERROR")                        # 语法错
    ]
    for c, v, expect in cases:
        try:
            r = safe_z3(c, v)
        except NotImplementedError as e:
            print(f"[TODO] {e}")
            break
        print(f"[{'OK' if expect in r else 'FAIL'}] {c[:45]} -> {r[:90]}")
