from sympy import Eq, solve
from .base_solver import BaseSolver

class EquationSolver(BaseSolver):
    def can_solve(self, expr) -> bool:
        return isinstance(expr, Eq)

    def solve(self, expr) -> list:
        steps = [f"Phương trình: {expr}"]
        x = list(expr.free_symbols)[0]
        sol = solve(expr, x)
        steps.append(f"Giải và tìm {x} = {sol}")
        return steps