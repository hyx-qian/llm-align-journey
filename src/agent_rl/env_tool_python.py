# -*- coding: utf-8 -*-
"""python（代码执行器）工具：受限 sandbox。

安全设计（四层）：
  ① AST 检查：禁 import 非白名单 / 禁危险内置名 / 禁 dunder 属性访问
  ② 受限 builtins（去掉 open/exec/eval/compile/__import__ 等）
  ③ 超时（SIGALRM，默认 5s）
  ④ 输出截断（500 字符）
返回格式与 safe_calc 一致：成功返回结果字符串，失败返回 "ERROR: ..."
"""
import ast
import builtins
import contextlib
import io
import signal

ALLOWED_MODULES = {"math", "re", "statistics", "itertools", "fractions", "decimal", "collections"}

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "input", "breakpoint", "exit", "quit", "vars", "dir", "getattr", "setattr",
    "delattr", "memoryview", "help", "license",
}


class UnsafeCode(Exception):
    pass


def _check_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_MODULES:
                    raise UnsafeCode(f"import '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod not in ALLOWED_MODULES:
                raise UnsafeCode(f"from '{mod}' import is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCode(f"name '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCode("dunder attribute access is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
            raise UnsafeCode(f"call '{node.func.id}' is not allowed")


def _timeout_handler(signum, frame):
    raise TimeoutError("execution timed out")


def safe_python(code: str, timeout: int = 5) -> str:
    """执行受限 Python 代码，返回 stdout（或最后一个表达式的值）。"""
    code = code.strip()
    if not code:
        return "ERROR: empty code"
    try:
        tree = ast.parse(code)
        _check_ast(tree)
    except UnsafeCode as e:
        return f"ERROR: {e}"
    except SyntaxError as e:
        return f"ERROR: {e.msg}"

    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in FORBIDDEN_NAMES}
    g = {"__builtins__": safe_builtins, "__name__": "__tool__"}
    buf = io.StringIO()

    # 若整体是单个表达式（无 print），求值后 repr
    is_expr = len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        with contextlib.redirect_stdout(buf):
            if is_expr:
                value = eval(compile(ast.Expression(tree.body[0].value), "<tool>", "eval"), g)
                if value is not None:
                    print(repr(value), file=buf)
            else:
                exec(compile(tree, "<tool>", "exec"), g)
    except TimeoutError:
        return f"ERROR: execution timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)

    out = buf.getvalue().strip()
    if not out:
        return "OK (no output)"
    return out[:500]


if __name__ == "__main__":
    cases = [
        ("print(sum(range(1,11)))", "55"),
        ("sum(range(10))", "45"),
        ("import os; print(os.listdir('.'))", "ERROR"),
        ("open('/etc/passwd').read()", "ERROR"),
        ("1/0", "ERROR"),
        ("print((lambda: [p for p in range(2,30) if all(p%d for d in range(2,p))])()[:5])", "primes"),
        ("__import__('os').system('ls')", "ERROR"),
        ("while True: pass", "ERROR"),
    ]
    for code, expect in cases:
        r = safe_python(code)
        ok = ("ERROR" in r) if expect == "ERROR" else (expect in r)
        print(f"[{'OK' if ok else 'FAIL'}] {code[:40]!r} -> {r[:80]!r}")
