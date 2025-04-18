from sympy.parsing.latex import parse_latex
from .arithmetic_solver import ArithmeticSolver
from .equation_solver import EquationSolver
from .calculus_solver import CalculusSolver

class MathSolver:
    def __init__(self):
        self.solvers = [
            EquationSolver(),
            CalculusSolver(),
            ArithmeticSolver()
        ]

    def solve_latex(self, latex_str: str) -> list:
        expr = parse_latex(latex_str)
        for solver in self.solvers:
            if solver.can_solve(expr):
                return solver.solve(expr)
        return ["Không thể giải biểu thức này."]
