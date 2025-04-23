import sympy as sp
import random
import json
import gc
from multiprocessing import Pool
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
from enum import Enum

class ProblemType(Enum):
    BASIC_ARITHMETIC = "basic_arithmetic"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    INEQUALITY = "inequality"
    SYSTEM = "system"
    DERIVATIVE = "derivative"
    EXPONENTIAL = "exponential"

class MathDatasetGenerator:
    """Class để tạo dataset toán học với các bài toán đa dạng và chuẩn hóa số."""

    def __init__(self):
        self.x, self.y = sp.symbols('x y')
        self.query_templates: Dict[ProblemType, List[str]] = {
            ProblemType.BASIC_ARITHMETIC: {
                '+': [
                    "Tính tổng của hai số",
                    "Cộng hai số, kết quả là bao nhiêu?",
                    "Tổng của {a} và {b} là gì?",
                    "Tìm giá trị của phép cộng"
                ],
                '-': [
                    "Tính hiệu của hai số",
                    "Trừ hai số, kết quả là bao nhiêu?",
                    "Hiệu của {a} và {b} là gì?",
                    "Tìm giá trị của phép trừ"
                ],
                '*': [
                    "Tính tích của hai số",
                    "Nhân hai số, kết quả là bao nhiêu?",
                    "Tích của {a} và {b} là gì?",
                    "Tìm giá trị của phép nhân"
                ],
                '/': [
                    "Tính thương của hai số",
                    "Chia hai số, kết quả là bao nhiêu?",
                    "Thương của {a} và {b} là gì?",
                    "Tìm giá trị của phép chia"
                ]
            },
            ProblemType.LINEAR: ["Giải tìm x", "Tìm x", "x = ?", "x bằng bao nhiêu?"],
            ProblemType.QUADRATIC: ["Giải phương trình", "Tìm nghiệm x", "x = ?", "Tìm các nghiệm"],
            ProblemType.INEQUALITY: ["Giải bất phương trình", "Tìm tập nghiệm", "x thỏa mãn điều kiện gì?"],
            ProblemType.SYSTEM: ["Giải hệ phương trình", "Tìm x và y", "(x, y) = ?"],
            ProblemType.DERIVATIVE: ["Tính đạo hàm", "Tìm f'(x)", "Đạo hàm của hàm số là gì?"],
            ProblemType.EXPONENTIAL: ["Giải phương trình mũ", "Tìm x", "x = ?"],
        }
        self.generators = {
            ProblemType.BASIC_ARITHMETIC: [
                self._generate_basic_addition,
                self._generate_basic_subtraction,
                self._generate_basic_multiplication,
                self._generate_basic_division,
                self._generate_mixed_expression,
                self._generate_nested_expression,
            ],
            ProblemType.LINEAR: [self._generate_linear_equation],
            # ProblemType.QUADRATIC: [self._generate_quadratic_equation],
            # ProblemType.INEQUALITY: [self._generate_inequality],
            # ProblemType.SYSTEM: [self._generate_system_of_equations],
            # ProblemType.DERIVATIVE: [self._generate_derivative],
            # ProblemType.EXPONENTIAL: [self._generate_exponential_equation],
        }
        self.weights = {
            ProblemType.BASIC_ARITHMETIC: 0.7,  # 70% cơ bản
            ProblemType.LINEAR: 0.3,
            # ProblemType.QUADRATIC: 0.08,
            # ProblemType.INEQUALITY: 0.05,
            # ProblemType.SYSTEM: 0.05,
            # ProblemType.DERIVATIVE: 0.03,
            # ProblemType.EXPONENTIAL: 0.04,
        }
        self.fixed_cases = [
            # Phép cộng
            {"eq": "1 + 1", "result": 2, "op": "+"},
            {"eq": "2 + 2", "result": 4, "op": "+"},
            {"eq": "3 + 3", "result": 6, "op": "+"},
            {"eq": "4 + 4", "result": 8, "op": "+"},
            {"eq": "5 + 5", "result": 10, "op": "+"},
            {"eq": "10 + 5", "result": 15, "op": "+"},
            {"eq": "8 + 7", "result": 15, "op": "+"},
            {"eq": "100 + 50", "result": 150, "op": "+"},
            {"eq": "0 + 0", "result": 0, "op": "+"},
            {"eq": "5 + 0", "result": 5, "op": "+"},
            {"eq": "0 + 7", "result": 7, "op": "+"},
            {"eq": "10 + 0", "result": 10, "op": "+"},
            {"eq": "-1 + 2", "result": 1, "op": "+"},
            {"eq": "3 + (-2)", "result": 1, "op": "+"},
            {"eq": "-2 + (-3)", "result": -5, "op": "+"},
            {"eq": "-5 + 5", "result": 0, "op": "+"},
            
            # Phép trừ
            {"eq": "2 - 1", "result": 1, "op": "-"},
            {"eq": "5 - 3", "result": 2, "op": "-"},
            {"eq": "10 - 4", "result": 6, "op": "-"},
            {"eq": "7 - 2", "result": 5, "op": "-"},
            {"eq": "15 - 5", "result": 10, "op": "-"},
            {"eq": "7 - 0", "result": 7, "op": "-"},
            {"eq": "0 - 5", "result": -5, "op": "-"},
            {"eq": "10 - 10", "result": 0, "op": "-"},
            {"eq": "0 - 0", "result": 0, "op": "-"},
            {"eq": "3 - (-2)", "result": 5, "op": "-"},
            {"eq": "-1 - 2", "result": -3, "op": "-"},
            {"eq": "2 - 5", "result": -3, "op": "-"},
            {"eq": "-3 - (-3)", "result": 0, "op": "-"},
            {"eq": "1 - 3", "result": -2, "op": "-"},
            {"eq": "2 - 7", "result": -5, "op": "-"},
            
            # Phép nhân
            {"eq": "2 \\times 2", "result": 4, "op": "\\times"},
            {"eq": "3 \\times 3", "result": 9, "op": "\\times"},
            {"eq": "4 \\times 5", "result": 20, "op": "\\times"},
            {"eq": "5 \\times 2", "result": 10, "op": "\\times"},
            {"eq": "10 \\times 10", "result": 100, "op": "\\times"},
            {"eq": "6 \\times 0", "result": 0, "op": "\\times"},
            {"eq": "0 \\times 8", "result": 0, "op": "\\times"},
            {"eq": "0 \\times 0", "result": 0, "op": "\\times"},
            {"eq": "7 \\times 1", "result": 7, "op": "\\times"},
            {"eq": "1 \\times 9", "result": 9, "op": "\\times"},
            {"eq": "1 \\times 1", "result": 1, "op": "\\times"},
            {"eq": "-2 \\times 3", "result": -6, "op": "\\times"},
            {"eq": "4 \\times (-2)", "result": -8, "op": "\\times"},
            {"eq": "-3 \\times (-3)", "result": 9, "op": "\\times"},
            
            # Phép chia
            {"eq": "4 \\div 2", "result": 2, "op": "\\div"},
            {"eq": "9 \\div 3", "result": 3, "op": "\\div"},
            {"eq": "15 \\div 5", "result": 3, "op": "\\div"},
            {"eq": "16 \\div 4", "result": 4, "op": "\\div"},
            {"eq": "25 \\div 5", "result": 5, "op": "\\div"},
            {"eq": "8 \\div 1", "result": 8, "op": "\\div"},
            {"eq": "1 \\div 1", "result": 1, "op": "\\div"},
            {"eq": "0 \\div 5", "result": 0, "op": "\\div"},
            {"eq": "0 \\div 10", "result": 0, "op": "\\div"},
            {"eq": "-6 \\div 2", "result": -3, "op": "\\div"},
            {"eq": "8 \\div (-4)", "result": -2, "op": "\\div"},
            {"eq": "-9 \\div (-3)", "result": 3, "op": "\\div"},
        ]

    def _random_query(self, problem_type: ProblemType, op: str = None,
                      a: Optional[int] = None, b: Optional[int] = None) -> str:
        if problem_type == ProblemType.BASIC_ARITHMETIC and op:
            query = random.choice(self.query_templates[problem_type][op])
            if a is not None and b is not None and "{a}" in query:
                return query.format(a=a, b=b)
            return query
        return random.choice(self.query_templates[problem_type])

    def _format_number(self, num) -> str:
        """Chuẩn hóa số thành chuỗi: số thập phân (làm tròn) hoặc phân số đơn giản."""
        try:
            # Convert input to SymPy object
            num = sp.sympify(num)
            if num.is_Integer:
                return str(num)
            if num.is_Rational:
                num_simp = sp.simplify(num)
                if num_simp.is_Integer:
                    return str(num_simp)
                num, den = num_simp.as_numer_denom()
                return f"\\frac{{{num}}}{{{den}}}"
            if num.is_Float:
                float_val = float(num)
                return f"{round(float_val, 4):.4f}".rstrip('0').rstrip('.')
            return sp.latex(num)
        except (TypeError, ValueError):
            return str(num)

    def _generate_basic_addition(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán cộng hai số."""
        try:
            a, b, _ = self._generate_numbers(level)
            result = a + b
            latex_eq = f"{a} + {b} ="
            steps = self._build_steps(latex_eq, f"{a} + {b} = {result}", result, detailed, op='+', a=a, b=b)
            return self._build_output(latex_eq, ProblemType.BASIC_ARITHMETIC, steps, op='+', a=a, b=b)
        except Exception:
            return None
    
    def _generate_basic_subtraction(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán trừ hai số."""        
        try:
            a, b, _ = self._generate_numbers(level)
            result = a - b
            latex_eq = f"{a} - {b} ="
            steps = self._build_steps(latex_eq, f"{a} - {b} = {result}", result, detailed, op='-', a=a, b=b)
            return self._build_output(latex_eq, ProblemType.BASIC_ARITHMETIC, steps, op='-', a=a, b=b)
        except Exception:
            return None

    def _generate_basic_multiplication(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán nhân hai số."""
        try:
            a, b, _ = self._generate_numbers(level, max_val=20)
            result = a * b
            latex_eq = f"{a} \\times {b} ="
            steps = self._build_steps(latex_eq, f"{a} \\times {b} = {result}", result, detailed, op='*', a=a, b=b)
            return self._build_output(latex_eq, ProblemType.BASIC_ARITHMETIC, steps, op='*', a=a, b=b)
        except Exception:
            return None

    def _generate_basic_division(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán chia hai số, đảm bảo chia hết."""
        try:
            b = random.randint(-20, 20)
            while b == 0:  # Tránh chia cho 0
                b = random.randint(-20, 20)
            result = random.randint(-20, 20)
            a = b * result
            latex_eq = f"{a} \\div {b} ="
            steps = self._build_steps(latex_eq, f"{a} \\div {b} = {result}", result, detailed, op='/', a=a, b=b)
            return self._build_output(latex_eq, ProblemType.BASIC_ARITHMETIC, steps, op='/', a=a, b=b)
        except Exception:
            return None
        
    def _generate_mixed_expression(self, detailed=True) -> Optional[Dict]:
        """Biểu thức nhiều bước không ngoặc: a + b * c - d"""
        try:
            a, b, c, d = [random.randint(1, 10) for _ in range(4)]
            expr = f"{a} + {b} * {c} - {d}"
            result = eval(expr)
            latex_expr = f"{a} + {b} \\times {c} - {d}"
            steps = [
                f"Biểu thức: \\({latex_expr}\\)",
                f"Theo thứ tự ưu tiên: \\({b} \\times {c} = {b * c}\\)",
                f"Tiếp theo: \\({a} + {b * c} = {a + b * c}\\)",
                f"Cuối cùng: \\({a + b * c} - {d} = {result}\\)"
            ]
            if not detailed:
                steps = [steps[0], steps[-1]]
            return self._build_output(latex_expr, ProblemType.BASIC_ARITHMETIC, steps)
        except Exception:
            return None
        
    def _generate_nested_expression(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán biểu thức số học với ngoặc, ví dụ: (a + b) * (c - d)."""
        try:
            a, b, c, d = [random.randint(1, 10) for _ in range(4)]
            op1 = random.choice(['+', '-'])
            op2 = random.choice(['+', '-', '*', '/'])
            latex_op_map = {'*': '\\times', '/': '\\div'}

            latex_expr = f"({a} {op1} {b}) {latex_op_map[op2]} ({c} - {d})"
            text_expr = f"({a} {op1} {b}) {op2} ({c} - {d})"

            expr = sp.simplify(text_expr)
            result = float(expr) if expr.is_number else sp.simplify(expr)

            val1 = a + b if op1 == '+' else a - b
            val2 = c - d
            
            steps = [
                f"Biểu thức: \\({latex_expr}\\)",
                f"Tính ngoặc 1: \\({a} {op1} {b} = {val1}\\)",
                f"Tính ngoặc 2: \\({c} - {d} = {val2}\\)",
                f"Thực hiện phép {op2}: \\({val1} {latex_op_map[op2]} {val2} = {self._format_number(result)}\\)"
            ]
            if not detailed:
                steps = [steps[0], steps[-1]]
            
            return self._build_output(latex_expr, ProblemType.BASIC_ARITHMETIC, steps)
        except Exception:
            return None

    def _generate_linear_equation(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        try:
            a, b, c = self._generate_numbers(level, max_val=10, non_zero=True)
            d = random.randint(-5, 5)
            k = random.randint(1, 5) if level != "easy" else 1
            form = random.choice(["standard", "distributive", "fraction", "subtraction"])

            latex_eq = None
            if form == "standard":  # ax + b = c
                eq = sp.Eq(a * self.x + b, c)
                latex_eq = f"{a} {sp.latex(self.x)} + {b} = {c}"
                steps = [
                    f"Phương trình: \\({sp.latex(eq)}\\)",
                    f"Chuyển vế: \\({sp.latex(a * self.x)} = {sp.latex(c - b)}\\)",
                    f"Chia hai vế cho {a}: \\(x = \\frac{{{sp.latex(c - b)}}}{{{a}}}\\)",
                    f"Nghiệm: \\(x = {self._format_number(sp.solve(eq, self.x)[0])}\\)"
                ]
            elif form == "distributive":  # a(x + d) = c
                eq = sp.Eq(a * (self.x + d), c)
                latex_eq = f"{a} ({sp.latex(self.x)} + {d}) = {c}"
                expanded = f"{sp.latex(a * self.x)} + {a} \\times {d}"
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Mở ngoặc: \\({expanded} = {c}\\)",
                    f"Tính toán: \\({sp.latex(a * self.x + a * d)} = {c}\\)",
                    f"Chuyển vế: \\({sp.latex(a * self.x)} = {sp.latex(c - a * d)}\\)",
                    f"Chia hai vế cho {a}: \\(x = \\frac{{{sp.latex(c - a * d)}}}{{{a}}}\\)",
                    f"Nghiệm: \\(x = {self._format_number(sp.solve(eq, self.x)[0])}\\)"
                ]
            elif form == "fraction":  # (ax + b)/k = c
                eq = sp.Eq((a * self.x + b) / k, c)
                latex_eq = f"\\frac{{{sp.latex(a * self.x + b)}}}{{{k}}} = {c}"
                expanded = f"{sp.latex(a * self.x + b)} = {c} \\times {k}"
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Nhân hai vế với {k}: \\({expanded}\\)",
                    f"Tính toán: \\({sp.latex(a * self.x + b)} = {c * k}\\)",
                    f"Chuyển vế: \\({sp.latex(a * self.x)} = {sp.latex(k * c - b)}\\)",
                    f"Chia hai vế cho {a}: \\(x = \\frac{{{sp.latex(k * c - b)}}}{{{a}}}\\)",
                    f"Nghiệm: \\(x = {self._format_number(sp.solve(eq, self.x)[0])}\\)"
                ]
            else:  # ax = c - b
                eq = sp.Eq(a * self.x, c - b)
                latex_eq = f"{a} {sp.latex(self.x)} = {c} - {b}"
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Rút gọn vế phải: \\({sp.latex(a * self.x)} = {c - b}\\)",
                    f"Chia hai vế cho {a}: \\(x = \\frac{{{c - b}}}{{{a}}}\\)",
                    f"Nghiệm: \\(x = {self._format_number(sp.solve(eq, self.x)[0])}\\)"
                ]

            if not detailed:
                steps = [steps[0], steps[-1]]
            return self._build_output(latex_eq, ProblemType.LINEAR, steps)
        except Exception:
            return None

    def _generate_quadratic_equation(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán phương trình bậc hai."""
        try:
            a, b, c = self._generate_numbers(level, max_val=5, non_zero=True)
            eq = sp.Eq(a * self.x**2 + b * self.x + c, 0)
            latex_eq = sp.latex(eq)
            solutions = sp.solve(eq, self.x)
            delta = b**2 - 4 * a * c
            steps = [
                f"Phương trình: \\({latex_eq}\\)",
                f"Delta: \\(\\Delta = {b}^2 - 4 \\cdot {a} \\cdot {c} = {delta}\\)"
            ]
            if detailed:
                steps.extend(self._quadratic_steps(a, b, delta, solutions))
            else:
                steps.append(f"Nghiệm: \\({' hoặc '.join(self._format_number(sol) for sol in solutions) if solutions else 'Không có nghiệm thực'}\\)")
            return self._build_output(latex_eq, ProblemType.QUADRATIC, steps)
        except Exception:
            return None

    def _generate_inequality(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán bất phương trình."""
        try:
            a, b, c = self._generate_numbers(level, max_val=10, non_zero=True)
            op = random.choice(['>', '<', '>=', '<='])
            ineq = {'>': a * self.x + b > c, '<': a * self.x + b < c,
                    '>=': a * self.x + b >= c, '<=': a * self.x + b <= c}[op]
            latex_ineq = sp.latex(ineq)
            solution = sp.solve(ineq, self.x)
            if not solution:
                return self._build_output(latex_ineq, ProblemType.INEQUALITY, [f"Bất phương trình: \\({latex_ineq}\\)", "Không có nghiệm"])
            
            solution_str = sp.latex(solution)
            steps = [
                f"Bất phương trình: \\({latex_ineq}\\)",
                f"Cách ly {a}x: \\({sp.latex(a * self.x)} {op} {sp.latex(c - b)}\\)",
                f"Chia cả hai vế cho {a}{' (đổi chiều bất phương trình)' if a < 0 else ''}: "
                f"\\(x {'<' if (op == '>' and a < 0) or (op == '<' and a > 0) else '>'} \\frac{{{sp.latex(c - b)}}}{{{a}}}\\)" if op in ['>', '<'] else
                f"\\(x {'<=' if (op == '>=' and a < 0) or (op == '<=' and a > 0) else '>='} \\frac{{{sp.latex(c - b)}}}{{{a}}}\\)",
                f"Tập nghiệm: \\({solution_str}\\)"
            ] if detailed else [f"Bất phương trình: \\({latex_ineq}\\)", f"Tập nghiệm: \\({solution_str}\\)"]
            return self._build_output(latex_ineq, ProblemType.INEQUALITY, steps)
        except Exception:
            return None

    def _generate_system_of_equations(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán hệ phương trình."""
        try:
            max_attempts = 10
            for _ in range(max_attempts):
                a1, b1, c1, a2, b2, c2 = [random.randint(-5, 5) for _ in range(6)]
                if a1 * b2 != a2 * b1:
                    break
            else:
                return None
            
            eq1 = sp.Eq(a1 * self.x + b1 * self.y, c1)
            eq2 = sp.Eq(a2 * self.x + b2 * self.y, c2)
            latex_eq = f"\\begin{{cases}} {sp.latex(eq1)} \\\\ {sp.latex(eq2)} \\end{{cases}}"
            solutions = sp.solve([eq1, eq2], (self.x, self.y))
            if not solutions:
                return self._build_output(latex_eq, ProblemType.SYSTEM, [f"Hệ phương trình: \\({latex_eq}\\)", "Không có nghiệm"])
            
            x_sol, y_sol = self._format_number(solutions[self.x]), self._format_number(solutions[self.y])
            steps = [
                f"Hệ phương trình: \\({latex_eq}\\)",
                "Sử dụng phương pháp thế hoặc cộng trừ",
                f"Tìm x: \\(x = {x_sol}\\)",
                f"Tìm y: \\(y = {y_sol}\\)",
                f"Nghiệm: \\((x, y) = ({x_sol}, {y_sol})\\)"
            ] if detailed else [f"Hệ phương trình: \\({latex_eq}\\)", f"Nghiệm: \\((x, y) = ({x_sol}, {y_sol})\\)"]
            return self._build_output(latex_eq, ProblemType.SYSTEM, steps)
        except Exception:
            return None

    def _generate_derivative(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán đạo hàm."""
        try:
            a, b, c = self._generate_numbers(level, max_val=5)
            func = a * self.x**2 + b * self.x + c
            latex_func = sp.latex(func)
            deriv = sp.diff(func, self.x)
            steps = [
                f"Hàm số: \\(f(x) = {latex_func}\\)",
                f"Đạo hàm: \\(f'(x) = {sp.latex(deriv)}\\)"
            ]
            if detailed:
                steps.insert(1, "Áp dụng quy tắc lấy đạo hàm")
                steps.extend([
                    f"Đạo hàm của \\({a}x^{{2}}\\): \\({2*a}x\\)",
                    f"Đạo hàm của {b}x: \\({b}\\)",
                    f"Đạo hàm của {c}: \\(0\\)"
                ])
            return self._build_output(latex_func, ProblemType.DERIVATIVE, steps)
        except Exception:
            return None

    def _generate_exponential_equation(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán phương trình mũ."""
        try:
            a, b = random.randint(1, 5), random.randint(1, 10)
            base = random.choice([2, 3, 5])
            eq = sp.Eq(a * base**self.x, b)
            latex_eq = sp.latex(eq)
            solutions = sp.solve(eq, self.x)
            if not solutions:
                return self._build_output(latex_eq, ProblemType.EXPONENTIAL, [f"Phương trình: \\({latex_eq}\\)", "Không có nghiệm"])
            
            solution_str = self._format_number(solutions[0])
            steps = [
                f"Phương trình: \\({latex_eq}\\)",
                f"Cách ly {base}^x: \\({base}^{{x}} = \\frac{{{b}}}{{{a}}}\\)",
                f"Áp dụng logarit: \\(x = \\log_{{{base}}} \\left( \\frac{{{b}}}{{{a}}} \\right)\\)",
                f"Nghiệm: \\(x = {solution_str}\\)"
            ] if detailed else [f"Phương trình: \\({latex_eq}\\)", f"Nghiệm: \\(x = {solution_str}\\)"]
            return self._build_output(latex_eq, ProblemType.EXPONENTIAL, steps)
        except Exception:
            return None
        
    def _generate_numbers(self, level: str, max_val: int = 50, 
                          non_zero: bool = False, allow_negative: bool = True) -> Tuple[int, ...]:
        """Tạo các số ngẫu nhiên dựa trên mức độ khó."""
        def get_number():
            if level == "easy":
                low, high = (-max_val, max_val) if allow_negative else (0, max_val)
                return random.randint(low, high)
            elif level == "medium":
                low, high = (-2 * max_val, 2 * max_val) if allow_negative else (0, 2 * max_val)
                return random.randint(low, high)
            elif level == "hard":
                return round(random.uniform(-max_val, max_val), 2) if allow_negative else round(random.uniform(1, max_val), 2)
            else:
                raise ValueError(f"Unknown level: {level}")
            
        numbers = [get_number() for _ in range(3)]
        
        if non_zero:
            for i in range(3):
                while numbers[i] == 0:
                    numbers[i] = get_number()

        return tuple(numbers)

    def _build_steps(self, latex_eq: str, calc_step: str, result: float, detailed: bool,
                     op: str, a: int, b: int) -> List[str]:
        """Xây dựng các bước giải cho bài toán cơ bản."""
        formatted_a = self._format_number(a)
        formatted_b = self._format_number(b)
        formatted_result = self._format_number(result)

        if detailed:
            if op == '+':
                if a < 0 or b < 0:
                    steps = [
                        f"Cộng với số âm: \\({formatted_a} + {formatted_b} = {formatted_result}\\)",
                        f"Kết quả phép cộng: \\({formatted_result}\\)"
                    ]
                else:
                    steps = [
                        f"Thực hiện phép cộng: \\({formatted_a} + {formatted_b} = {formatted_result}\\)",
                        f"Kết quả: \\({formatted_result}\\)"
                    ]
            elif op == '-':
                if a < 0 or b < 0:
                    steps = [
                        f"Trừ với số âm: \\({formatted_a} - {formatted_b} = {formatted_result}\\)",
                        f"Kết quả phép trừ: \\({formatted_result}\\)"
                    ]
                else:
                    steps = [
                        f"Thực hiện phép trừ: \\({formatted_a} - {formatted_b} = {formatted_result}\\)",
                        f"Kết quả: \\({formatted_result}\\)"
                    ]
            elif op == '*':
                if a < 0 or b < 0:
                    steps = [
                        f"Nhân với số âm: \\({formatted_a} \\times {formatted_b} = {formatted_result}\\)",
                        f"Kết quả phép nhân: \\({formatted_result}\\)"
                    ]
                else:
                    steps = [
                        f"Thực hiện phép nhân: \\({formatted_a} \\times {formatted_b} = {formatted_result}\\)",
                        f"Kết quả: \\({formatted_result}\\)"
                    ]
            else:  # /
                steps = [
                    f"Thực hiện phép chia: \\({formatted_a} \\div {formatted_b} = {formatted_result}\\)",
                    f"Kết quả: \\({formatted_result}\\)"
                ]
        else:
            steps = [f"Bài toán: \\({latex_eq}\\)", f"Kết quả: \\({formatted_result}\\)"]
        return steps

    def _build_output(self, latex_eq: str, problem_type: ProblemType, steps: List[str],
                      op: str = None, a: Optional[int] = None, b: Optional[int] = None) -> Dict:
        """Tạo cấu trúc dữ liệu đầu ra thống nhất."""
        return {
            "problem_type": problem_type.value,
            "latex_equation": latex_eq,
            "query": self._random_query(problem_type, op, a, b),
            "solution_steps": steps
        }
    
    def _generate_fixed_basic_cases(self, detailed: bool = True, case_idx: int = None) -> Optional[Dict]:
        """Tạo bài toán từ danh sách fixed cases, sử dụng chỉ số case_idx."""
        if case_idx is None or case_idx >= len(self.fixed_cases):
            return None
        case = self.fixed_cases[case_idx]
        latex_eq = f"{case['eq']} ="
        op = case['op'].replace('\\times', '*').replace('\\div', '/')

        try:
            # Chuyển biểu thức LaTeX sang dạng text để sympy phân tích
            expr_text = case['eq'].replace('\\times', '*').replace('\\div', '/')
            expr = sp.sympify(expr_text, evaluate=False)
            if isinstance(expr, (sp.Add, sp.Mul)):
                args = expr.args
                a = args[0]
                if op == '+':
                    b = args[1]
                elif op == '-':
                    b = -args[1]
                elif op == '*':
                    b = args[1]
                else:  # op == '/'
                    # Phép chia: args[1] là x**(-1), lấy nghịch đảo để được b
                    if isinstance(args[1], sp.Pow) and args[1].args[1] == -1:
                        b = args[1].args[0]  # Lấy số bị chia
                    else:
                        b = args[1]
            else:
                # Trường hợp lồng nhau, ví dụ: (2 + 3) * 2
                a = sp.sympify(expr_text.split(op)[0].strip())
                b = sp.sympify(expr_text.split(op)[1].strip())
        except Exception:
            # Fallback: Trích xuất thủ công nếu sympy thất bại
            parts = expr_text.split(op)
            a = sp.sympify(parts[0].strip())
            b = sp.sympify(parts[1].strip())

        calc_step = f"{case['eq']} = {self._format_number(case['result'])}"
        steps = self._build_steps(latex_eq, calc_step, case['result'], detailed, op=op, a=a, b=b)
        return self._build_output(latex_eq, ProblemType.BASIC_ARITHMETIC, steps, op=op, a=a, b=b)

    def _quadratic_steps(self, a: int, b: int, delta: int, solutions: List[sp.Expr]) -> List[str]:
        """Tạo các bước giải chi tiết cho phương trình bậc hai."""
        if delta > 0:
            sol1, sol2 = self._format_number(solutions[0]), self._format_number(solutions[1])
            return [
                f"Delta dương, có hai nghiệm thực",
                f"Nghiệm: \\(x = \\frac{{-{b} \\pm \\sqrt{{{delta}}}}}{{2 \\cdot {a}}}\\)",
                f"Nghiệm thứ nhất: \\(x_1 = {sol1}\\)",
                f"Nghiệm thứ hai: \\(x_2 = {sol2}\\)"
            ]
        elif delta == 0:
            sol = self._format_number(solutions[0])
            return [
                f"Delta bằng 0, có nghiệm kép",
                f"Nghiệm: \\(x = \\frac{{-{b}}}{{2 \\cdot {a}}}\\)",
                f"Nghiệm: \\(x = {sol}\\)"
            ]
        return ["Delta âm, không có nghiệm thực"]

    def generate_sample(self, sample_idx: int) -> Optional[Dict]:
        try:
            problem_type = random.choices(list(self.weights.keys()), list(self.weights.values()), k=1)[0]
            generator = random.choice(self.generators[problem_type])
            level = random.choice(["easy", "medium"])
            detailed = random.random() > 0.3
            sample = generator(detailed=detailed, level=level)
            gc.collect()
            return sample
        except Exception:
            return None

    def generate_dataset(self, num_samples: int, output_file: str, batch_size: int = 100, max_attempts: int = 40000) -> None:
        dataset = []
        seen_problems = set()  # Lưu trữ các bài toán đã sinh

        # Thêm fixed cases chỉ một lần
        for i in range(min(len(self.fixed_cases), num_samples)):
            sample = self._generate_fixed_basic_cases(detailed=True, case_idx=i)
            if sample and sample["latex_equation"] not in seen_problems:
                dataset.append(sample)
                seen_problems.add(sample["latex_equation"])
        
        print(f"Added {len(dataset)} fixed cases")

        # Sinh các mẫu ngẫu nhiên cho đến khi đạt đủ num_samples bài duy nhất
        remaining = num_samples - len(dataset)
        attempts = 0
        with Pool() as pool:
            while len(dataset) < num_samples and attempts < max_attempts:
                batch_samples = pool.map(self.generate_sample, range(batch_size))
                
                for sample in batch_samples:
                    if sample and sample["latex_equation"] not in seen_problems:
                        dataset.append(sample)
                        seen_problems.add(sample["latex_equation"])
                        if len(dataset) >= num_samples:
                            break
                
                attempts += batch_size
                print(f"Generated {len(dataset)} unique problems after {attempts} attempts")

                # Lưu checkpoint mỗi 1000 bài
                # if len(dataset) % 1000 == 0:
                #     with open(output_file + ".checkpoint", 'w', encoding='utf-8') as f:
                #         json.dump(dataset[:num_samples], f, ensure_ascii=False, indent=2)

        # Cắt dataset nếu vượt quá num_samples
        dataset = dataset[:num_samples]
        
        # Lưu file cuối cùng
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Final dataset: {len(dataset)} unique problems")

def validate_dataset(file_path: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        problems = [sample["latex_equation"] for sample in dataset]
        unique_problems = set(problems)
        print(f"Total problems: {len(problems)}")
        print(f"Unique problems: {len(unique_problems)}")
        if len(unique_problems) < len(problems):
            print(f"Found {len(problems) - len(unique_problems)} duplicates")

if __name__ == "__main__":
    generator = MathDatasetGenerator()
    generator.generate_dataset(18000, "data/mathsolver/math_dataset.json", batch_size=100)

    validate_dataset("data/mathsolver/math_dataset.json")