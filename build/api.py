from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from src.mathwriting.scripts.detector import LatexDetector

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

detector = LatexDetector(
    vocab_file="src/mathwriting/checkpoints/vocab.txt",
    checkpoint_path="src/mathwriting/checkpoints/best_model (10).pt"
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
        latex = detector.predict_from_image(image)
        return jsonify({"latex": latex})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "MathSolverAI image-to-LaTeX API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)