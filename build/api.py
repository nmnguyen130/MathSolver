from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64

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
        
    elif request.is_json:
        data = request.get_json()
        if "image_base64" not in data:
            return jsonify({"error": "Missing 'image_base64' in JSON body"}), 400
        try:
            img_bytes = base64.b64decode(data["image_base64"])
            image = Image.open(BytesIO(img_bytes))
        except Exception as e:
            return jsonify({"error": "Invalid base64 image", "detail": str(e)}), 400
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