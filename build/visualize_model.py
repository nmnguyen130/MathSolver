from PIL import Image

from src.mathwriting.scripts.detector import LatexDetector

if __name__ == "__main__":
    detector = LatexDetector(
        data_dir="data/mathwriting-2024",
        checkpoint_path="src/mathwriting/checkpoints/best_model.pt"
    )
    image = Image.open("000119.png")
    detector.visualize_processing(image)