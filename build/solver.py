from src.mathsolver.scripts import MathSolver

def main():
    solver = MathSolver()
    test_cases = [
        r"3 + 5",
        r"x + 3 = 7",
        r"\frac{d}{dx}(x^2)",
        r"\int x^2 \, dx"
    ]
    for latex in test_cases:
        print(f"\nKiểm tra: {latex}")
        steps = solver.solve_latex(latex)
        print("\n".join(steps))

if __name__ == "__main__":
    main()