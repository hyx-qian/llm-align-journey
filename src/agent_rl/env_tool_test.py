# -*- coding: utf-8 -*-
"""test_runner：跑 pytest 验证代码（v9 代码场景 T2 的新工具）。

【主任务】代码场景 T1={code_runner} → T2={code_runner, test_runner}
  test_runner 是"不可替代"的工具：测试通过率不能心算，必须真跑。

接口（与 safe_calc/safe_z3 一致，返回字符串）：
  safe_test(code: str, tests: str, timeout: int = 10) -> str
  · code  = "def add(a,b): return a+b"
  · tests = "assert add(1,2)==3\\nassert add(0,0)==0"
  · 返回："PASS 2/2" / "FAIL: AssertionError ..." / "ERROR: ..."

安全：临时目录 + subprocess + 超时 + 危险模式黑名单。
"""
import os
import re
import subprocess
import sys
import tempfile

MAX_CODE = 4000
MAX_TESTS = 2000
MAX_PASS_LINES = 10  # 最多解析 10 条 assert

_FORBIDDEN = [
    "import os", "import subprocess", "import sys", "import shutil",
    "__import__", "open(", "eval(", "exec(", "rmtree", "socket",
]


def _check_forbidden(text: str) -> str:
    for bad in _FORBIDDEN:
        if bad in text:
            return bad
    return ""


def safe_test(code: str, tests: str, timeout: int = 10) -> str:
    """执行 code 并跑 tests（每行一个 assert），返回通过情况。"""
    code = (code or "").strip()
    tests = (tests or "").strip()
    if not code or not tests:
        return "ERROR: empty code or tests"
    if len(code) > MAX_CODE:
        return f"ERROR: code too long (>{MAX_CODE})"
    if len(tests) > MAX_TESTS:
        return f"ERROR: tests too long (>{MAX_TESTS})"

    hit = _check_forbidden(code) or _check_forbidden(tests)
    if hit:
        return f"ERROR: forbidden pattern '{hit}'"

    test_lines = [ln.strip() for ln in tests.split("\n") if ln.strip()][:MAX_PASS_LINES]
    if not test_lines:
        return "ERROR: no assert lines"
    for ln in test_lines:
        if not ln.startswith("assert"):
            return f"ERROR: test line must start with 'assert': {ln[:40]}"

    with tempfile.TemporaryDirectory() as d:
        sol = os.path.join(d, "solution.py")
        tst = os.path.join(d, "test_solution.py")
        with open(sol, "w", encoding="utf-8") as f:
            f.write(code)
        # 生成 pytest 测试：每条 assert 一个 test_ 函数
        funcs = []
        for i, ln in enumerate(test_lines):
            funcs.append(f"def test_case_{i}():\n    {ln}\n")
        header = f"import sys\nsys.path.insert(0, {d!r})\nfrom solution import *\n\n"
        with open(tst, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(funcs))

        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", tst, "-q", "--no-header", "-p", "no:cacheprovider"],
                capture_output=True, text=True, timeout=timeout, cwd=d,
            )
            out = (r.stdout or "") + (r.stderr or "")
            total = len(test_lines)
            if r.returncode == 0:
                return f"PASS {total}/{total}"
            m = re.search(r"(\d+) failed", out)
            n_fail = int(m.group(1)) if m else total
            n_pass = max(0, total - n_fail)
            # 取第一处失败原因
            err = ""
            for line in out.split("\n"):
                if "AssertionError" in line or "Error" in line:
                    err = line.strip()[:120]
                    break
            return f"FAIL {n_pass}/{total}: {err}"
        except subprocess.TimeoutExpired:
            return f"ERROR: timeout (>{timeout}s)"
        except FileNotFoundError:
            return "ERROR: pytest not installed"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    cases = [
        ("def add(a,b): return a+b", "assert add(1,2)==3\nassert add(0,0)==0", "PASS"),
        ("def add(a,b): return a-b", "assert add(1,2)==3", "FAIL"),
        ("def f(): return 1", "assert f()==2", "FAIL"),
        ("import os", "assert True", "ERROR"),
        ("def f(): return 1", "print('x')", "ERROR"),
        ("def f(): return 1", "", "ERROR"),
    ]
    for code, tests, expect in cases:
        r = safe_test(code, tests)
        ok = expect in r
        print(f"[{'OK' if ok else 'FAIL'}] {code[:26]:<26} | {tests[:30]:<30} -> {r[:60]}")
