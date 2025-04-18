from sympy import Derivative, Integral, diff, integrate
from .base_solver import BaseSolver

class CalculusSolver(BaseSolver):
    def can_solve(self, expr) -> bool:
        return isinstance(expr, (Derivative, Integral))

    def solve(self, expr) -> list:
        steps = [f"Biểu thức giải tích: {expr}"]
        if isinstance(expr, Derivative):
            result = diff(expr.expr, *expr.variables)
            steps.append(f"Đạo hàm: {result}")
        elif isinstance(expr, Integral):
            result = integrate(expr.function, *expr.variables)
            steps.append(f"Tích phân: {result}")
        return steps