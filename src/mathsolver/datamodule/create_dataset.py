import sympy as sp
import random
import json
import gc
from multiprocessing import Pool
from typing import Dict, Optional, List
from tqdm import tqdm

class MathDatasetGenerator:
    """Class để tạo dataset toán học với các bài toán đa dạng và chuẩn hóa số."""

    def __init__(self):
        self.x = sp.Symbol('x')
        self.y = sp.Symbol('y')
        self.query_map = {
            "linear": [
                "giải tìm x", "tìm x", "x = ?", "x bằng bao nhiêu?",
                "solve for x", "find x", "what is x?"
            ],
            "quadratic": [
                "giải phương trình", "tìm nghiệm x", "x = ?", "tìm các nghiệm",
                "solve the equation", "find the roots", "what are the solutions?"
            ],
            "inequality": [
                "giải bất phương trình", "tìm tập nghiệm", "x thỏa mãn điều kiện gì?",
                "solve the inequality", "find the solution set"
            ],
            "system": [
                "giải hệ phương trình", "tìm x và y", "(x, y) = ?",
                "solve the system", "find x and y", "what is (x, y)?"
            ],
            "trigonometric": [
                "giải phương trình lượng giác", "tìm x", "x = ?",
                "solve the trigonometric equation", "find x"
            ],
            "derivative": [
                "tính đạo hàm", "tìm f'(x)", "đạo hàm của hàm số là gì?",
                "compute the derivative", "find the derivative", "what is f'(x)?"
            ],
            "integral": [
                "tính tích phân", "tìm nguyên hàm", "tích phân là gì?",
                "compute the integral", "find the antiderivative"
            ],
            "logarithmic": [
                "giải phương trình logarit", "tìm x", "x = ?",
                "solve the logarithmic equation", "find x"
            ],
            "exponential": [
                "giải phương trình mũ", "tìm x", "x = ?",
                "solve the exponential equation", "find x"
            ],
            "definite_integral": [
                "tính tích phân xác định", "tìm giá trị tích phân", "tích phân từ a đến b là bao nhiêu?",
                "compute the definite integral", "find the value of the integral"
            ]
        }

    def _random_query(self, problem_type: str) -> str:
        """Chọn ngẫu nhiên một query từ danh sách tương ứng với loại bài toán."""
        return random.choice(self.query_map[problem_type])

    def _format_number(self, num: sp.Expr) -> str:
        """Chuẩn hóa số thành chuỗi: số thập phân (làm tròn) hoặc phân số đơn giản."""
        if num.is_Rational:
            num_simp = sp.simplify(num)
            if num_simp.is_Integer:
                return str(num_simp)
            num, den = num_simp.as_numer_denom()
            return f"\\frac{{{num}}}{{{den}}}"
        try:
            float_val = float(num)
            return f"{round(float_val, 4):.4f}".rstrip('0').rstrip('.')
        except (TypeError, ValueError):
            return sp.latex(num)

    def _generate_linear_equation(self, detailed: bool = True) -> Optional[Dict]:
        try:
            a = random.randint(-10, 10)
            b = random.randint(-10, 10)
            c = random.randint(-10, 10)
            while a == 0:
                a = random.randint(-10, 10)
            
            eq = sp.Eq(a * self.x + b, c)
            latex_eq = sp.latex(eq)
            
            solutions = sp.solve(eq, self.x)
            if not solutions:
                return {
                    "latex_equation": latex_eq,
                    "query": self._random_query("linear"),
                    "solution_steps": [
                        f"Phương trình cho trước: \\({latex_eq}\\)",
                        "Phương trình này không có nghiệm."
                    ]
                }
            
            solution_str = self._format_number(solutions[0])
            if detailed:
                steps = [
                    f"Phương trình cho trước: \\({latex_eq}\\)",
                    f"Cách ly {a}x: \\({sp.latex(a * self.x)} = {sp.latex(c - b)}\\)",
                    f"Chia cả hai vế cho {a}: \\(x = \\frac{{{sp.latex(c - b)}}}{{{a}}}\\)",
                    f"Nghiệm: \\(x = {solution_str}\\)"
                ]
            else:
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Nghiệm: \\(x = {solution_str}\\)"
                ]
            
            return {
                "latex_equation": latex_eq,
                "query": self._random_query("linear"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def _generate_quadratic_equation(self, detailed: bool = True) -> Optional[Dict]:
        try:
            a = random.randint(-5, 5)
            b = random.randint(-10, 10)
            c = random.randint(-10, 10)
            while a == 0:
                a = random.randint(-5, 5)
            
            eq = sp.Eq(a * self.x**2 + b * self.x + c, 0)
            latex_eq = sp.latex(eq)
            
            solutions = sp.solve(eq, self.x)
            delta = b**2 - 4 * a * c
            if detailed:
                steps = [
                    f"Phương trình bậc hai cho trước: \\({latex_eq}\\)",
                    f"Sử dụng công thức nghiệm: \\(x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}\\)",
                    f"Tính delta: \\(\\Delta = {b}^2 - 4 \\cdot {a} \\cdot {c} = {delta}\\)"
                ]
                if delta > 0:
                    sqrt_delta = sp.sqrt(delta)
                    sol1 = self._format_number(solutions[0])
                    sol2 = self._format_number(solutions[1])
                    steps.extend([
                        f"Delta dương, phương trình có hai nghiệm thực",
                        f"Nghiệm: \\(x = \\frac{{-{b} \\pm \\sqrt{{{delta}}}}}{{2 \\cdot {a}}}\\)",
                        f"Nghiệm thứ nhất: \\(x_1 = {sol1}\\)",
                        f"Nghiệm thứ hai: \\(x_2 = {sol2}\\)"
                    ])
                elif delta == 0:
                    sol = self._format_number(solutions[0])
                    steps.extend([
                        f"Delta bằng 0, phương trình có nghiệm kép",
                        f"Nghiệm: \\(x = \\frac{{-{b}}}{{2 \\cdot {a}}}\\)",
                        f"Nghiệm: \\(x = {sol}\\)"
                    ])
                else:
                    steps.extend([
                        f"Delta âm, phương trình không có nghiệm thực",
                        "Không có nghiệm thực"
                    ])
            else:
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Delta: \\(\\Delta = {delta}\\)",
                    f"Nghiệm: \\({' hoặc '.join(self._format_number(sol) for sol in solutions) if solutions else 'Không có nghiệm thực'}\\)"
                ]
            
            return {
                "latex_equation": latex_eq,
                "query": self._random_query("quadratic"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def _generate_inequality(self, detailed: bool = True) -> Optional[Dict]:
        try:
            a = random.randint(-10, 10)
            b = random.randint(-10, 10)
            c = random.randint(-10, 10)
            while a == 0:
                a = random.randint(-10, 10)
            
            op = random.choice(['>', '<', '>=', '<='])
            if op == '>':
                ineq = a * self.x + b > c
            elif op == '<':
                ineq = a * self.x + b < c
            elif op == '>=':
                ineq = a * self.x + b >= c
            else:
                ineq = a * self.x + b <= c
            
            latex_ineq = sp.latex(ineq)
            solution = sp.solve(ineq, self.x)
            if not solution:
                return {
                    "latex_equation": latex_ineq,
                    "query": self._random_query("inequality"),
                    "solution_steps": [
                        f"Bất phương trình cho trước: \\({latex_ineq}\\)",
                        "Bất phương trình không có nghiệm"
                    ]
                }
            
            solution_str = sp.latex(solution)
            if detailed:
                steps = [
                    f"Bất phương trình cho trước: \\({latex_ineq}\\)",
                    f"Cách ly {a}x: \\({sp.latex(a * self.x)} {op} {sp.latex(c - b)}\\)",
                    f"Chia cả hai vế cho {a}{' (đổi chiều bất phương trình)' if a < 0 else ''}: "
                    f"\\(x {'<' if (op == '>' and a < 0) or (op == '<' and a > 0) else '>'}"
                    f" \\frac{{{sp.latex(c - b)}}}{{{a}}}\\)" if op in ['>', '<'] else
                    f"\\(x {'<=' if (op == '>=' and a < 0) or (op == '<=' and a > 0) else '>='}"
                    f" \\frac{{{sp.latex(c - b)}}}{{{a}}}\\)",
                    f"Tập nghiệm: \\({solution_str}\\)"
                ]
            else:
                steps = [
                    f"Bất phương trình: \\({latex_ineq}\\)",
                    f"Tập nghiệm: \\({solution_str}\\)"
                ]
            
            return {
                "latex_equation": latex_ineq,
                "query": self._random_query("inequality"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def _generate_system_of_equations(self, detailed: bool = True) -> Optional[Dict]:
        try:
            max_attempts = 10
            attempt = 0
            
            while attempt < max_attempts:
                a1 = random.randint(-5, 5)
                b1 = random.randint(-5, 5)
                c1 = random.randint(-10, 10)
                a2 = random.randint(-5, 5)
                b2 = random.randint(-5, 5)
                c2 = random.randint(-10, 10)
                
                if a1 * b2 != a2 * b1:
                    break
                attempt += 1
            
            if attempt >= max_attempts:
                return None
            
            eq1 = sp.Eq(a1 * self.x + b1 * self.y, c1)
            eq2 = sp.Eq(a2 * self.x + b2 * self.y, c2)
            latex_eq = f"\\begin{{cases}} {sp.latex(eq1)} \\\\ {sp.latex(eq2)} \\end{{cases}}"
            
            solutions = sp.solve([eq1, eq2], (self.x, self.y))
            if not solutions:
                return {
                    "latex_equation": latex_eq,
                    "query": self._random_query("system"),
                    "solution_steps": [
                        f"Hệ phương trình cho trước: \\({latex_eq}\\)",
                        "Hệ phương trình không có nghiệm"
                    ]
                }
            
            x_sol = self._format_number(solutions[self.x])
            y_sol = self._format_number(solutions[self.y])
            if detailed:
                steps = [
                    f"Hệ phương trình cho trước: \\({latex_eq}\\)",
                    "Sử dụng phương pháp thế hoặc cộng trừ để giải",
                    "Nhân các phương trình nếu cần để căn chỉnh hệ số",
                    f"Tìm x: \\(x = {x_sol}\\)",
                    f"Tìm y: \\(y = {y_sol}\\)",
                    f"Nghiệm: \\((x, y) = ({x_sol}, {y_sol})\\)"
                ]
            else:
                steps = [
                    f"Hệ phương trình: \\({latex_eq}\\)",
                    f"Nghiệm: \\((x, y) = ({x_sol}, {y_sol})\\)"
                ]
            
            return {
                "latex_equation": latex_eq,
                "query": self._random_query("system"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def _generate_trigonometric_equation(self, detailed: bool = True) -> Optional[Dict]:
        try:
            a = round(random.uniform(-0.9, 0.9), 4)
            func = random.choice([sp.sin, sp.cos])
            eq = sp.Eq(func(self.x), a)
            latex_eq = sp.latex(eq)
            
            solutions = sp.solve(eq, self.x)[:2]
            if not solutions:
                return {
                    "latex_equation": latex_eq,
                    "query": self._random_query("trigonometric"),
                    "solution_steps": [
                        f"Phương trình cho trước: \\({latex_eq}\\)",
                        "Không có nghiệm trong phạm vi yêu cầu"
                    ]
                }
            
            sol1 = self._format_number(solutions[0])
            sol2 = self._format_number(solutions[1]) if len(solutions) > 1 else sol1
            if detailed:
                steps = [
                    f"Phương trình cho trước: \\({latex_eq}\\)",
                    f"Tìm góc x sao cho {func.__name__}(x) = {a:.4f}",
                    f"Sử dụng hàm ngược {func.__name__}",
                    f"Nghiệm: \\({sol1}, {sol2}\\)"
                ]
            else:
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Nghiệm: \\({sol1}, {sol2}\\)"
                ]
            
            return {
                "latex_equation": latex_eq,
                "query": self._random_query("trigonometric"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def _generate_derivative(self, detailed: bool = True) -> Optional[Dict]:
        try:
            a = random.randint(-5, 5)
            b = random.randint(-5, 5)
            c = random.randint(-5, 5)
            func = a * self.x**2 + b * self.x + c
            latex_func = sp.latex(func)
            
            deriv = sp.diff(func, self.x)
            if detailed:
                steps = [
                    f"Hàm số cho trước: \\(f(x) = {latex_func}\\)",
                    "Áp dụng quy tắc lấy đạo hàm cho từng hạng tử",
                    f"Đạo hàm của {a}x²: \\({2*a}x\\)",
                    f"Đạo hàm của {b}x: \\({b}\\)",
                    f"Đạo hàm của {c}: \\(0\\)",
                    f"Kết quả: \\(f'(x) = {sp.latex(deriv)}\\)"
                ]
            else:
                steps = [
                    f"Hàm số: \\(f(x) = {latex_func}\\)",
                    f"Đạo hàm: \\(f'(x) = {sp.latex(deriv)}\\)"
                ]
            
            return {
                "latex_equation": latex_func,
                "query": self._random_query("derivative"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def _generate_integral(self, detailed: bool = True) -> Optional[Dict]:
        try:
            a = random.randint(-5, 5)
            b = random.randint(-5, 5)
            func = a * self.x**2 + b * self.x
            latex_func = sp.latex(func)
            
            integral = sp.integrate(func, self.x)
            if detailed:
                steps = [
                    f"Tích phân cho trước: \\(\\int ({latex_func}) \\, dx\\)",
                    "Tính tích phân từng hạng tử riêng lẻ",
                    f"Tích phân của {a}x²: \\({self._format_number(a * self.x**3 / 3)}\\)",
                    f"Tích phân của {b}x: \\({self._format_number(b * self.x**2 / 2)}\\)",
                    "Cộng hằng số tích phân C",
                    f"Kết quả: \\({sp.latex(integral)} + C\\)"
                ]
            else:
                steps = [
                    f"Tích phân: \\(\\int ({latex_func}) \\, dx\\)",
                    f"Kết quả: \\({sp.latex(integral)} + C\\)"
                ]
            
            return {
                "latex_equation": latex_func,
                "query": self._random_query("integral"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def _generate_logarithmic_equation(self, detailed: bool = True) -> Optional[Dict]:
        try:
            a = random.randint(1, 5)
            b = random.randint(1, 10)
            base = random.choice([2, 3, 10])
            eq = sp.Eq(sp.log(a * self.x, base), b)
            latex_eq = sp.latex(eq)
            
            solutions = sp.solve(eq, self.x)
            if not solutions:
                return {
                    "latex_equation": latex_eq,
                    "query": self._random_query("logarithmic"),
                    "solution_steps": [
                        f"Phương trình cho trước: \\({latex_eq}\\)",
                        "Phương trình không có nghiệm."
                    ]
                }
            
            solution_str = self._format_number(solutions[0])
            if detailed:
                steps = [
                    f"Phương trình cho trước: \\({latex_eq}\\)",
                    f"Chuyển sang dạng mũ: \\({base}^{{{b}}} = {a}x\\)",
                    f"Cách ly x: \\(x = \\frac{{{base}^{{{b}}}}}{{{a}}}\\)",
                    f"Nghiệm: \\(x = {solution_str}\\)"
                ]
            else:
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Nghiệm: \\(x = {solution_str}\\)"
                ]
            
            return {
                "latex_equation": latex_eq,
                "query": self._random_query("logarithmic"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def _generate_exponential_equation(self, detailed: bool = True) -> Optional[Dict]:
        try:
            a = random.randint(1, 5)
            b = random.randint(1, 10)
            base = random.choice([2, 3, 5])
            eq = sp.Eq(a * base**self.x, b)
            latex_eq = sp.latex(eq)
            
            solutions = sp.solve(eq, self.x)
            if not solutions:
                return {
                    "latex_equation": latex_eq,
                    "query": self._random_query("exponential"),
                    "solution_steps": [
                        f"Phương trình cho trước: \\({latex_eq}\\)",
                        "Phương trình không có nghiệm."
                    ]
                }
            
            solution_str = self._format_number(solutions[0])
            if detailed:
                steps = [
                    f"Phương trình cho trước: \\({latex_eq}\\)",
                    f"Cách ly {base}^x: \\({base}^x = \\frac{{{b}}}{{{a}}}\\)",
                    f"Áp dụng logarit: \\(x = \\log_{{{base}}} \\left( \\frac{{{b}}}{{{a}}} \\right)\\)",
                    f"Nghiệm: \\(x = {solution_str}\\)"
                ]
            else:
                steps = [
                    f"Phương trình: \\({latex_eq}\\)",
                    f"Nghiệm: \\(x = {solution_str}\\)"
                ]
            
            return {
                "latex_equation": latex_eq,
                "query": self._random_query("exponential"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def _generate_definite_integral(self, detailed: bool = True) -> Optional[Dict]:
        try:
            a = random.randint(-5, 5)
            b = random.randint(-5, 5)
            lower = random.randint(-5, 0)
            upper = random.randint(1, 5)
            func = a * self.x + b
            latex_func = sp.latex(func)
            
            integral = sp.integrate(func, (self.x, lower, upper))
            antideriv = sp.integrate(func, self.x)
            integral_str = self._format_number(integral)
            if detailed:
                steps = [
                    f"Tích phân xác định: \\(\\int_{{{lower}}}^{{{upper}}} ({latex_func}) \\, dx\\)",
                    f"Tìm nguyên hàm: \\(\\int ({latex_func}) \\, dx = {sp.latex(antideriv)} + C\\)",
                    f"Áp dụng công thức tích phân xác định: \\([ {sp.latex(antideriv)} ]_{{{lower}}}^{{{upper}}}\\)",
                    f"Thay giới hạn: \\({self._format_number(antideriv.subs(self.x, upper))} - {self._format_number(antideriv.subs(self.x, lower))}\\)",
                    f"Kết quả: \\({integral_str}\\)"
                ]
            else:
                steps = [
                    f"Tích phân: \\(\\int_{{{lower}}}^{{{upper}}} ({latex_func}) \\, dx\\)",
                    f"Kết quả: \\({integral_str}\\)"
                ]
            
            return {
                "latex_equation": latex_func,
                "query": self._random_query("definite_integral"),
                "solution_steps": steps
            }
        except Exception:
            return None

    def generate_sample(self, sample_idx: int) -> Optional[Dict]:
        try:
            generators = [
                (self._generate_linear_equation, "linear"),
                (self._generate_quadratic_equation, "quadratic"),
                (self._generate_inequality, "inequality"),
                (self._generate_system_of_equations, "system"),
                (self._generate_trigonometric_equation, "trigonometric"),
                (self._generate_derivative, "derivative"),
                (self._generate_integral, "integral"),
                (self._generate_logarithmic_equation, "logarithmic"),
                (self._generate_exponential_equation, "exponential"),
                (self._generate_definite_integral, "definite_integral")
            ]
            generator, problem_type = random.choice(generators)
            detailed = random.random() > 0.3  # 70% chi tiết, 30% ngắn gọn
            sample = generator(detailed=detailed)
            gc.collect()
            return sample
        except Exception:
            return None

    def generate_dataset(self, num_samples: int, output_file: str, batch_size: int = 500) -> None:
        dataset = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with Pool() as pool:
            for batch_idx in tqdm(range(num_batches), desc="Generating dataset"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_samples = pool.map(self.generate_sample, range(start_idx, end_idx))
                
                batch_samples = [s for s in batch_samples if s is not None]
                dataset.extend(batch_samples)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generator = MathDatasetGenerator()
    generator.generate_dataset(5000, "data/mathsolver/math_dataset.json", batch_size=100)