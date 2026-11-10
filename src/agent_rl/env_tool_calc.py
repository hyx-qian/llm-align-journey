"""Safe calculator tool executor with AST whitelist parsing.

设计要点：只允许 数字/四则运算/幂/取模/负号/白名单函数，拒绝任意代码执行——
sandbox 安全的教科书实现（对应 JD 加分项"sandbox 环境构建"）。
"""
import ast
import operator

ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
ALLOWED_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": lambda x: x ** 0.5,
}


def safe_calc(expression: str) -> float:
    """Evaluate a math expression safely (AST whitelist, no arbitrary code)."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
        return ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARYOPS:
        return ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS:
            args = [_eval_node(a) for a in node.args]
            return ALLOWED_FUNCS[node.func.id](*args)
        raise ValueError(f"disallowed function call")
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


if __name__ == "__main__":
    # 自测：6 个用例（含一个攻击用例应被拒绝）
    cases = ["2 + 3 * 4", "(2+3)*4/5", "2**10", "sqrt(144)", "max(3, 7, 5)", "__import__('os').system('ls')"]
    for c in cases:
        try:
            print(f"{c!r:45s} -> {safe_calc(c)}")
        except Exception as e:
            print(f"{c!r:45s} -> REJECTED ({type(e).__name__})")
