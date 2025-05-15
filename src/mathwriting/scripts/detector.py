import cv2
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

from src.mathwriting.preprocessing.tokenizer import LaTeXTokenizer
from src.mathwriting.datamodule.transforms import get_val_test_transform
from src.mathwriting.models.model import MathWritingModel

class LatexDetector:
    def __init__(self, vocab_file: str, checkpoint_path: str, img_size=(224, 224), device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        vocab_path = Path(vocab_file)
        self.tokenizer = LaTeXTokenizer(vocab_path)

        self.model = MathWritingModel(
            vocab_size=self.tokenizer.vocab_size,
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.mean = 0.7931
        self.std = 0.1738

        self.transform = get_val_test_transform(img_size, self.mean, self.std)

    def _enhance_image(self, image: Image.Image, save_path: str = None) -> Image.Image:
        """Tăng cường ảnh: làm nét, trắng nền, nét đen rõ ràng (cho ảnh từ camera)."""
        # Convert to numpy for OpenCV processing
        img_np = np.array(image.convert("RGB"))

        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Làm mờ nhẹ để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Dùng threshold Otsu để tách nền và nét
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dùng morphology để loại bỏ chấm nhỏ và làm nét vùng viết
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Tạo nền trắng
        result = np.ones_like(img_np) * 255  # RGB trắng

        # Đặt pixel đen cho vùng có nét
        for c in range(3):
            result[:, :, c][cleaned > 0] = 0

        # Convert lại thành ảnh PIL
        enhanced_image = Image.fromarray(result)

        # Tăng độ nét một chút (tùy chọn)
        enhanced_image = enhanced_image.filter(ImageFilter.SHARPEN)

        # (Tùy chọn) Tăng độ tương phản
        enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(1.2)

        # Lưu nếu cần
        if save_path:
            enhanced_image.save(save_path)
            print(f"Enhanced image saved at: {save_path}")

        return enhanced_image
    
    def _preprocess_image(self, image: Image.Image, save_path: str = 'preprocessed_image.png'):
        # Apply enhancement
        image = self._enhance_image(image, save_path="enhanced_image.png")

        # Apply transform
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        img_tensor = augmented["image"].float().unsqueeze(0)

        if save_path:
            # Unnormalize
            mean = torch.tensor([self.mean] * 3).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([self.std] * 3).view(1, 3, 1, 1).to(self.device)
            unnormalized = img_tensor * std + mean
            unnormalized = unnormalized.clamp(0, 1)

            # Convert to PIL image
            img_np = (unnormalized[0].cpu().numpy() * 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
            Image.fromarray(img_np).save(save_path)
            print(f"Preprocessed image saved at: {save_path}")

        return img_tensor
    
    def predict_from_image(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        img_tensor = self._preprocess_image(image).to(self.device)

        with torch.no_grad():
            preds = self.model.generate(img_tensor)
            preds_decoded = [self.tokenizer.decode(pred.tolist()) for pred in preds]
        return preds_decoded[0]