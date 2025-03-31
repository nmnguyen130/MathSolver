import torch
from shared.preprocessing.inkml_loader import InkMLDatasetLoader
# from src.mathwriting.preprocessing.bounding_box import BoundingBoxExtractor
from shared.preprocessing.symbol_extractor import SymbolExtractor
from src.mathwriting.models.model import Seq2Seq

class FormulaRecognizer:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = Seq2Seq().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # self.bbox_extractor = BoundingBoxExtractor()
        self.symbol_extractor = SymbolExtractor()

    def preprocess_image(self, image_path):
        # Extract bounding boxes and strokes from image
        bounding_boxes = self.bbox_extractor.extract(image_path)
        strokes = self.symbol_extractor.extract_strokes(bounding_boxes)

        # Convert strokes to tensor (x, y, stroke_indicator)
        input_tensor = InkMLDatasetLoader.normalize_and_pad(strokes)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        return input_tensor

    def decode_prediction(self, prediction):
        # Assuming a vocab dictionary exists for decoding
        vocab = InkMLDatasetLoader.get_vocab()
        tokens = [vocab[idx] for idx in prediction]
        return ''.join(tokens).replace('<pad>', '').replace('<eos>', '')

    def recognize(self, image_path):
        input_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(input_tensor)

        predicted_indices = torch.argmax(output, dim=-1).squeeze().cpu().numpy()
        formula = self.decode_prediction(predicted_indices)
        return formula

if __name__ == "__main__":
    recognizer = FormulaRecognizer(model_path="path/to/your/model.pth")
    result = recognizer.recognize("path/to/your/image.png")
    print(f"Recognized formula: {result}")