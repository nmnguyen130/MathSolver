from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from src.mathwriting.scripts.detector import LatexDetector
from src.mathsolver.scripts.solver import MathSolver

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

latex_detector = LatexDetector(
    vocab_file="src/mathwriting/checkpoints/vocab.txt",
    checkpoint_path="src/mathwriting/checkpoints/epoch_04.pt"
)

math_solver = MathSolver(
    vocab_file="src/mathsolver/checkpoints/vocab.txt",
    checkpoint_path="src/mathsolver/checkpoints/epoch_29.pt"
)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    # Dạng multipart/form-data (form upload file)
    image_file = request.files["image"]

    try:
        image = Image.open(image_file)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    try:
        equation = latex_detector.predict_from_image(image)
        return jsonify({"equation": equation})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route("/solve", methods=["POST"])
def solve():
    if "equation" not in request.form or "query" not in request.form:
        return jsonify({"error": "Missing equation or query"}), 400
    
    equation = request.form["equation"]
    query = request.form["query"]
    print(equation, query)
    
    try:
        solution = math_solver.solve(equation, query)
        return jsonify({"solution": solution})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "MathSolverAI image-to-LaTeX API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)