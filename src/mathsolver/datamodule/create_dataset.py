import sympy as sp
import random
import json
import gc
from multiprocessing import Pool
from typing import Dict, Optional, List, Tuple
from enum import Enum

class ProblemType(Enum):
    BASIC = "basic"
    LINEAR = "linear"

class MathDatasetGenerator:
    def __init__(self):
        self.x, self.y = sp.symbols('x y')
        self.query_templates: Dict[ProblemType, List[str]] = {
            ProblemType.BASIC: {
                '+': [
                    "Tính tổng của hai số {a} và {b}",
                    "Cộng {a} với {b}, kết quả là bao nhiêu?",
                    "Tổng của {a} và {b} là gì?",
                    "Thực hiện phép cộng {a} + {b}"
                ],
                '-': [
                    "Tính hiệu của {a} và {b}",
                    "Trừ {b} từ {a}, kết quả là bao nhiêu?",
                    "Hiệu của {a} và {b} là gì?",
                    "Thực hiện phép trừ {a} - {b}"
                ],
                '*': [
                    "Tính tích của {a} và {b}",
                    "Nhân {a} với {b}, kết quả là bao nhiêu?",
                    "Tích của {a} và {b} là gì?",
                    "Thực hiện phép nhân {a} * {b}"
                ],
                '/': [
                    "Tính thương của {a} và {b}",
                    "Chia {a} cho {b}, kết quả là bao nhiêu?",
                    "Thương của {a} và {b} là gì?",
                    "Thực hiện phép chia {a} / {b}"
                ]
            },
            ProblemType.LINEAR: [
                "Giải phương trình {latex_eq}",
                "Tìm x sao cho {latex_eq}",
                "Xác định giá trị của x trong {latex_eq}",
                "Tìm nghiệm của {latex_eq}"
            ]
        }
        self.generators = {
            ProblemType.BASIC: [
                self._generate_basic_addition,
                self._generate_basic_subtraction,
                self._generate_basic_multiplication,
                self._generate_basic_division,
                self._generate_mixed_expression,
                self._generate_nested_expression,
            ],
            ProblemType.LINEAR: [self._generate_linear_equation],
        }
        self.weights = {
            # ProblemType.BASIC: 1,
            ProblemType.LINEAR: 1,
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
                      a: Optional[int] = None, b: Optional[int] = None, latex_eq: str = None) -> str:
        if problem_type == ProblemType.BASIC and op:
            query = random.choice(self.query_templates[problem_type][op])
            return query.format(a=a, b=b)
        elif problem_type == ProblemType.LINEAR and latex_eq:
            query = random.choice(self.query_templates[problem_type])
            return query.format(latex_eq=latex_eq)
        return "Tính giá trị biểu thức"

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
                return f"{round(float_val, 2):.2f}".rstrip('0').rstrip('.')
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
            return self._build_output(latex_eq, ProblemType.BASIC, steps, self._format_number(result), op='+', a=a, b=b)
        except Exception:
            return None
    
    def _generate_basic_subtraction(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán trừ hai số."""        
        try:
            a, b, _ = self._generate_numbers(level)
            result = a - b
            latex_eq = f"{a} - {b} ="
            steps = self._build_steps(latex_eq, f"{a} - {b} = {result}", result, detailed, op='-', a=a, b=b)
            return self._build_output(latex_eq, ProblemType.BASIC, steps, self._format_number(result), op='-', a=a, b=b)
        except Exception:
            return None

    def _generate_basic_multiplication(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán nhân hai số."""
        try:
            a, b, _ = self._generate_numbers(level, max_val=20)
            result = a * b
            latex_eq = f"{a} \\times {b} ="
            steps = self._build_steps(latex_eq, f"{a} \\times {b} = {result}", result, detailed, op='*', a=a, b=b)
            return self._build_output(latex_eq, ProblemType.BASIC, steps, self._format_number(result), op='*', a=a, b=b)
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
            return self._build_output(latex_eq, ProblemType.BASIC, steps, self._format_number(result), op='/', a=a, b=b)
        except Exception:
            return None
        
    def _generate_mixed_expression(self, detailed=True) -> Optional[Dict]:
        """Biểu thức nhiều bước không ngoặc: a + b * c - d"""
        try:
            a, b, c, d = [random.randint(1, 10) for _ in range(4)]
            result = a + b * c - d
            latex_expr = f"{a} + {b} \\times {c} - {d}"
            steps = [
                f"Biểu thức: \\({latex_expr}\\)",
                f"Thực hiện phép nhân trước: \\({b} \\times {c} = {b * c}\\)",
                f"Thực hiện phép cộng: \\({a} + {b * c} = {a + b * c}\\)",
                f"Thực hiện phép trừ: \\({a + b * c} - {d} = {result}\\)"
            ]
            if not detailed:
                steps = [steps[0], steps[-1]]
            return self._build_output(latex_expr, ProblemType.BASIC, steps, self._format_number(result))
        except Exception:
            return None
        
    def _generate_nested_expression(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        """Tạo bài toán biểu thức số học với ngoặc, ví dụ: (a + b) * (c - d)."""
        try:
            a, b, c, d = [random.randint(1, 10) for _ in range(4)]
            op1 = random.choice(['+', '-'])
            op2 = random.choice(['+', '-', '*', '/'])
            latex_op_map = {'*': '\\times', '/': '\\div'}

            latex_expr = f"({a} {op1} {b}) {latex_op_map.get(op2, op2)} ({c} - {d})"
            val1 = a + b if op1 == '+' else a - b
            val2 = c - d
            if op2 == '+':
                result = val1 + val2
            elif op2 == '-':
                result = val1 - val2
            elif op2 == '*':
                result = val1 * val2
            else:  # /
                if val2 == 0:
                    return None  # Tránh chia cho 0
                result = val1 / val2

            steps = [
                f"Biểu thức: \\({latex_expr}\\)",
                f"Tính ngoặc đầu tiên: \\({a} {op1} {b} = {val1}\\)",
                f"Tính ngoặc thứ hai: \\({c} - {d} = {val2}\\)",
                f"Thực hiện phép {op2}: \\({val1} {latex_op_map.get(op2, op2)} {val2} = {self._format_number(result)}\\)"
            ]
            if not detailed:
                steps = [steps[0], steps[-1]]
            return self._build_output(latex_expr, ProblemType.BASIC, steps, self._format_number(result))
        except Exception:
            return None

    def _generate_linear_equation(self, detailed: bool = True, level: str = "easy") -> Optional[Dict]:
        try:
            a, b, c = self._generate_numbers(level, max_val=10, non_zero=True)
            d = random.randint(-5, 5)
            k = random.randint(1, 5) if level != "easy" else 1
            form = random.choice(["standard", "distributive", "fraction", "subtraction", "no_solution", "infinite_solutions", "both_sides"])

            latex_eq = None
            answer = None
            steps = []
            if form == "standard":  # ax + b = c
                eq = sp.Eq(a * self.x + b, c)
                latex_eq = f"{a}x + {b} = {c}"
                solution = sp.solve(eq, self.x)
                answer = f"x = {self._format_number(solution[0])}" if solution else "Không có nghiệm"
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    random.choice([
                        f"Trừ {b} hai vế: \\({a}x + {b} - {b} = {c} - {b}\\)",
                        f"Chuyển {b} sang vế phải: \\({a}x = {c} - {b}\\)",
                    ]),
                    f"Rút gọn: \\({a}x = {c - b}\\)",
                    f"Chia hai vế cho {a}: \\(x = \\frac{{{c - b}}}{{{a}}}\\)",
                    f"Kết quả: \\({answer}\\)"
                ]
            elif form == "distributive":  # a(x + d) = c
                eq = sp.Eq(a * (self.x + d), c)
                latex_eq = f"{a}(x + {d}) = {c}"
                solution = sp.solve(eq, self.x)
                answer = f"x = {self._format_number(solution[0])}" if solution else "Không có nghiệm"
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Mở ngoặc biểu thức: \\({a}(x + {d}) = {a}x + {a} \\times {d} = {a}x + {a * d}\\)",
                    f"Phương trình sau khi mở ngoặc: \\({a}x + {a * d} = {c}\\)",
                    random.choice([
                        f"Trừ {a * d} hai vế: \\({a}x + {a * d} - {a * d} = {c} - {a * d}\\)",
                        f"Chuyển {a * d} sang vế phải: \\({a}x = {c} - {a * d}\\)",
                    ]),
                    f"Rút gọn: \\({a}x = {c - a * d}\\)",
                    f"Chia hai vế cho {a}: \\(x = \\frac{{{c - a * d}}}{{{a}}}\\)",
                    f"Kết quả: \\({answer}\\)"
                ]
            elif form == "fraction":  # (ax + b)/k = c
                eq = sp.Eq((a * self.x + b) / k, c)
                latex_eq = f"\\frac{{{a}x + {b}}}{{{k}}} = {c}"
                solution = sp.solve(eq, self.x)
                answer = f"x = {self._format_number(solution[0])}" if solution else "Không có nghiệm"
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Nhân cả hai vế với {k}: \\({k} \\times \\frac{{{a}x + {b}}}{{{k}}} = {k} \\times {c}\\)",
                    f"Rút gọn: \\({a}x + {b} = {c * k}\\)",
                    random.choice([
                        f"Trừ {b} hai vế: \\({a}x + {b} - {b} = {c * k} - {b}\\)",
                        f"Chuyển {b} sang vế phải: \\({a}x = {c * k} - {b}\\)",
                    ]),
                    f"Rút gọn: \\({a}x = {c * k - b}\\)",
                    f"Chia hai vế cho {a}: \\(x = \\frac{{{c * k - b}}}{{{a}}}\\)",
                    f"Kết quả: \\({answer}\\)"
                ]
            elif form == "subtraction":  # ax = c - b
                eq = sp.Eq(a * self.x, c - b)
                latex_eq = f"{a}x = {c} - {b}"
                solution = sp.solve(eq, self.x)
                answer = f"x = {self._format_number(solution[0])}" if solution else "Không có nghiệm"
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Tính vế phải: \\({c} - {b} = {c - b}\\)",
                    f"Phương trình sau khi rút gọn: \\({a}x = {c - b}\\)",
                    f"Chia hai vế cho {a}: \\(x = \\frac{{{c - b}}}{{{a}}}\\)",
                    f"Kết quả: \\({answer}\\)"
                ]
            elif form == "no_solution":  # ax + b = ax + c, b != c
                eq = sp.Eq(a * self.x + b, a * self.x + c)
                latex_eq = f"{a}x + {b} = {a}x + {c}"
                answer = "Không có nghiệm"
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    random.choice([
                        f"Trừ \\({a}x\\) hai vế: \\({a}x + {b} - {a}x = {a}x + {c} - {a}x\\)",
                        f"Chuyển \\({a}x\\) sang vế phải: \\({b} = {a}x + {c} - {a}x\\)",
                    ]),
                    f"Rút gọn: \\({b} = {c}\\)",
                    f"Kết luận: Vì \\({b} \\neq {c}\\), phương trình không có nghiệm",
                    f"Kết quả: \\({answer}\\)"
                ]
            elif form == "infinite_solutions":  # ax + b = ax + b
                eq = sp.Eq(a * self.x + b, a * self.x + b)
                latex_eq = f"{a}x + {b} = {a}x + {b}"
                answer = "Vô số nghiệm"
                choice = random.choice([
                    [
                        f"Trừ hai vế cho \\({a}x + {b}\\): \\({a}x + {b} - ({a}x + {b}) = {a}x + {b} - ({a}x + {b})\\)",
                        f"Mở ngoặc: \\({a}x + {b} - {a}x - {b} = {a}x + {b} - {a}x - {b}\\)"
                    ],
                    [
                        f"Chuyển \\({a}x + {b}\\) sang vế phải: \\(0 = {a}x + {b} - ({a}x + {b})\\)",
                        f"Mở ngoặc: \\(0 = {a}x + {b} - {a}x - {b}\\)"
                    ],
                ])
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    *choice,
                    f"Rút gọn: \\(0 = 0\\)",
                    f"Vì phương trình đúng với mọi giá trị của \\(x\\), nên nó có vô số nghiệm",
                    f"Kết quả: \\({answer}\\)"
                ]
            elif form == "both_sides":  # ax + b = cx + d
                c2 = random.randint(-10, 10)  # Hệ số cho x ở vế phải
                while c2 == a:  # Tránh trường hợp vô số nghiệm
                    c2 = random.randint(-10, 10)
                eq = sp.Eq(a * self.x + b, c2 * self.x + d)
                latex_eq = f"{a}x + {b} = {c2}x + {d}"
                solution = sp.solve(eq, self.x)
                answer = f"x = {self._format_number(solution[0])}" if solution else "Không có nghiệm"
                choice = random.choice([
                    [
                        f"Trừ hai vế cho \\({c2}x\\): \\({a}x + {b} - {c2}x = {c2}x + {d} - {c2}x\\)",
                        f"Rút gọn: \\({a - c2}x + {b} = {d}\\)",
                        f"Trừ hai vế cho {b}: \\({a - c2}x + {b} - {b} = {d} - {b}\\)",
                    ],
                    f"Chuyển \\({c2}x\\) sang vế trái và \\({b}\\) sang vế phải: \\({a}x - {c2}x = {d} - {b}\\)",
                ])
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    *choice,
                    f"Rút gọn: \\({a - c2}x = {d - b}\\)",
                    f"Chia hai vế cho {a - c2}: \\(x = \\frac{{{d - b}}}{{{a - c2}}}\\)",
                    f"Kết quả: \\({answer}\\)"
                ]

            if not detailed:
                steps = [steps[0], steps[-1]]
            return self._build_output(latex_eq, ProblemType.LINEAR, steps, answer, c=c)
        except Exception:
            return None

    def _generate_numbers(self, level: str, max_val: int = 50, 
                          non_zero: bool = False, allow_negative: bool = True) -> Tuple[int, ...]:
        def get_number():
            if level == "easy":
                low, high = (-max_val, max_val) if allow_negative else (1, max_val)
                return random.randint(low, high)
            elif level == "medium":
                low, high = (-2 * max_val, 2 * max_val) if allow_negative else (1, 2 * max_val)
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
                steps = [
                    f"Thực hiện phép cộng: \\({formatted_a} + {formatted_b} = {formatted_result}\\)",
                    f"Kết quả: \\({formatted_result}\\)"
                ]
            elif op == '-':
                steps = [
                    f"Thực hiện phép trừ: \\({formatted_a} - {formatted_b} = {formatted_result}\\)",
                    f"Kết quả: \\({formatted_result}\\)"
                ]
            elif op == '*':
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
            steps = [f"Tính: \\({latex_eq} {formatted_result}\\)", f"Kết quả: \\({formatted_result}\\)"]
        return steps

    def _build_output(self, latex_eq: str, problem_type: ProblemType, steps: List[str], answer: str,
                       op: str = None, a: Optional[int] = None, b: Optional[int] = None, c: Optional[int] = None) -> Dict:
        query = self._random_query(problem_type, op, a, b, latex_eq)
        return {
            "problem_type": problem_type.value,
            "latex_equation": latex_eq,
            "query": query,
            "solution_steps": steps,
            "answer": answer
        }
    
    def _generate_fixed_basic_cases(self, detailed: bool = True, case_idx: int = None) -> Optional[Dict]:
        """Tạo bài toán từ danh sách fixed cases, sử dụng chỉ số case_idx."""
        if case_idx is None or case_idx >= len(self.fixed_cases):
            return None
        case = self.fixed_cases[case_idx]
        latex_eq = f"{case['eq']} ="
        op = case['op'].replace('\\times', '*').replace('\\div', '/')

        try:
            # Chuyển biểu thức LaTeX sang dạng text
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
                    if isinstance(args[1], sp.Pow) and args[1].args[1] == -1:
                        b = args[1].args[0]  # Lấy số bị chia
                    else:
                        b = args[1]
            else:
                # Fallback: Tách thủ công cho các trường hợp phức tạp
                parts = expr_text.split(op, 1)  # Tách tại toán tử đầu tiên
                a = sp.sympify(parts[0].strip())
                b = sp.sympify(parts[1].strip())
                if op == '-':
                    b = -b  # Điều chỉnh cho phép trừ

            calc_step = f"{case['eq']} = {self._format_number(case['result'])}"
            steps = self._build_steps(latex_eq, calc_step, case['result'], detailed, op=op, a=a, b=b)
            return self._build_output(
                latex_eq=latex_eq,
                problem_type=ProblemType.BASIC,
                steps=steps,
                answer=self._format_number(case['result']),
                op=op,
                a=a,
                b=b
            )
        except Exception as e:
            print(f"Error in fixed case {case['eq']}: {e}")
            return None

    def generate_sample(self, sample_idx: int) -> Optional[Dict]:
        try:
            problem_type = random.choices(list(self.weights.keys()), list(self.weights.values()), k=1)[0]
            generator = random.choice(self.generators[problem_type])
            level = random.choice(["easy", "medium"])
            detailed = random.random() > 0.1
            sample = generator(detailed=detailed, level=level)
            gc.collect()
            return sample
        except Exception:
            return None

    def generate_dataset(self, num_samples: int, output_file: str, batch_size: int = 100, max_attempts: int = 100000) -> None:
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

        # Cắt dataset nếu vượt quá num_samples
        dataset = dataset[:num_samples]
        
        # Lưu file cuối cùng
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Final dataset: {len(dataset)} unique problems")

    def validate_dataset(self, file_path: str) -> None:
            """Kiểm tra dataset: trùng lặp và tính đúng đắn."""
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            problems = [sample["latex_equation"] for sample in dataset]
            unique_problems = set(problems)
            print(f"Total problems: {len(problems)}")
            print(f"Unique problems: {len(unique_problems)}")
            if len(unique_problems) < len(problems):
                print(f"Found {len(problems) - len(unique_problems)} duplicates")

            # Kiểm tra tính đúng đắn của đáp án
            errors = 0
            for sample in dataset:
                if sample["problem_type"] == "linear":
                    try:
                        eq = sp.sympify(sample["latex_equation"].replace('=', '-'), evaluate=False)
                        solution = sp.solve(eq, self.x)
                        expected_answer = f"x = {self._format_number(solution[0])}" if solution else "Không có nghiệm"
                        if sample["answer"] not in [expected_answer, "Không có nghiệm", "Vô số nghiệm"]:
                            errors += 1
                            print(f"Error in {sample['latex_equation']}: Expected {expected_answer}, got {sample['answer']}")
                    except Exception:
                        continue
            print(f"Found {errors} answer errors")

if __name__ == "__main__":
    generator = MathDatasetGenerator()
    generator.generate_dataset(16057, "data/mathsolver/math_linear_dataset.json", batch_size=100)
    generator.validate_dataset("data/mathsolver/math_linear_dataset.json")