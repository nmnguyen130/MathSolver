from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from src.mathwriting.scripts.detector import LatexDetector

app = Flask(__name__)
CORS(app)

detector = LatexDetector(
    data_dir="data/mathwriting-2024",
    checkpoint_path="src/mathwriting/checkpoints/best_model.pt"
)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" in request.files:
        # Dạng multipart/form-data (form upload file)
        image = Image.open(request.files["image"])
        
    else:
        return jsonify({"error": "No image provided"}), 400

    try:
        latex = detector.predict_from_image(image)
        return jsonify({"latex": latex})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "MathSolverAI image-to-LaTeX API is running!"

if __name__ == "__main__":
    app.run(debug=True)