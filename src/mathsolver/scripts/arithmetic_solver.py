from sympy import Expr
from sympy.parsing.latex import parse_latex
from .base_solver import BaseSolver

class ArithmeticSolver(BaseSolver):
    def can_solve(self, expr: Expr) -> bool:
        return expr.is_number or expr.is_Add or expr.is_Mul

    def solve(self, expr: Expr) -> list:
        steps = [f"Biểu thức ban đầu: {expr}"]
        result = expr.evalf()
        steps.append(f"Tính toán kết quả: {result}")
        return steps