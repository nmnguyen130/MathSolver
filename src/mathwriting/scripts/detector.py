import cv2
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.mathwriting.models.model import MathWritingModel
from src.mathwriting.preprocessing.tokenizer import LaTeXTokenizer

class LatexDetector:
    def __init__(self, vocab_file: str, checkpoint_path: str, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        vocab_path = Path(vocab_file)
        self.tokenizer = LaTeXTokenizer(vocab_path)

        self.model = MathWritingModel(
            vocab_size=self.tokenizer.vocab_size,
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = A.Compose([
            A.ToGray(p=1.0),
            A.Normalize(mean=(0.7931, 0.7931, 0.7931), std=(0.1738, 0.1738, 0.1738)),
            ToTensorV2()
        ])

    def _enhance_image(self, image: Image.Image, save_path: str = None) -> Image.Image:
        """Tăng cường ảnh: làm nét, trắng nền, nét đen rõ ràng (cho ảnh từ camera)."""
        if save_path:
            original_path = save_path.replace(".png", "_original.png")
            image.save(original_path)
            print(f"Original image saved at: {original_path}")

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
    
    def _preprocess_image(self, image: Image.Image, img_size=(224, 224), save_path: str = 'preprocessed_image.png'):
        max_height, max_width = img_size
        mean = 0.7931
        std = 0.1738
        bg_value = (1.0 - mean) / std

        # Apply enhancement
        image = self._enhance_image(image, save_path="enhanced_image.png")

        # Apply transform
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        img_tensor = augmented["image"].float()
        _, img_h, img_w = img_tensor.size()

        # Tạo tensor nền trắng
        src = torch.full((1, 3, max_height, max_width), bg_value, dtype=img_tensor.dtype)

        # Resize nếu ảnh lớn hơn max_height hoặc max_width
        if img_h > max_height or img_w > max_width:
            scale = min(max_height / img_h, max_width / img_w)
            new_h = int(img_h * scale)
            new_w = int(img_w * scale)
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w), 
                           mode='bicubic', align_corners=False).squeeze(0)
        else:
            new_h, new_w = img_h, img_w  # Giữ nguyên kích thước nếu ảnh nhỏ

        # Căn giữa và pad ảnh
        pad_h_start = (max_height - new_h) // 2
        pad_w_start = (max_width - new_w) // 2
        pad_h_end = pad_h_start + new_h
        pad_w_end = pad_w_start + new_w
        src[0, :, pad_h_start:pad_h_end, pad_w_start:pad_w_end] = img_tensor

        if save_path:
            # Unnormalize
            mean = torch.tensor([0.7931, 0.7931, 0.7931]).view(3, 1, 1)
            std = torch.tensor([0.1738, 0.1738, 0.1738]).view(3, 1, 1)
            unnormalized = src[0] * std + mean
            unnormalized = unnormalized.clamp(0, 1)

            # Convert to PIL image
            img_np = (unnormalized.cpu().numpy() * 255).astype(np.uint8)  # shape: (3, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
            pil_image = Image.fromarray(img_np)
            pil_image.save(save_path)
            print(f"Preprocessed image saved at: {save_path}")

        return src
    
    def predict_from_image(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        img_tensor = self._preprocess_image(image).to(self.device)

        with torch.no_grad():
            preds = self.model.generate(img_tensor)
            preds_decoded = [self.tokenizer.decode(pred.tolist()) for pred in preds]
        return preds_decoded[0]