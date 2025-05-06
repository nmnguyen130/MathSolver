from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter

from src.mathwriting.models.model import MathWritingModel
from src.mathwriting.preprocessing.tokenizer import LaTeXTokenizer

class LatexDetector:
    def __init__(self, vocab_file: str, checkpoint_path: str, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        vocab_path = Path(vocab_file)
        self.tokenizer = LaTeXTokenizer(vocab_path)

        # Load mô hình
        self.model = MathWritingModel(
            vocab_size=self.tokenizer.vocab_size,
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738))
        ])

    def predict_from_image(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        img_tensor = self._preprocess_image(image).to(self.device)

        with torch.no_grad():
            preds = self.model.generate(img_tensor)
            preds_decoded = [self.tokenizer.decode(pred.tolist()) for pred in preds]
        return preds_decoded[0]
    
    def _preprocess_image(self, image: Image.Image, img_size=(224, 224)):
        max_height, max_width = img_size
        bg_value = 1.0  # Giá trị nền trắng sau chuẩn hóa

        img_tensor = self.transform(image)

        _, img_h, img_w = img_tensor.size()

        # Tạo tensor nền trắng
        src = torch.full((1, 3, max_height, max_width), bg_value, dtype=img_tensor.dtype)

        # Resize nếu ảnh lớn hơn max_height hoặc max_width
        if img_h > max_height or img_w > max_width:
            scale = min(max_height / img_h, max_width / img_w)
            new_h = int(img_h * scale)
            new_w = int(img_w * scale)
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w), 
                                    mode='bilinear', align_corners=False).squeeze(0)
        else:
            new_h, new_w = img_h, img_w  # Giữ nguyên kích thước nếu ảnh nhỏ

        # Căn giữa và pad ảnh
        pad_h_start = (max_height - new_h) // 2
        pad_w_start = (max_width - new_w) // 2
        pad_h_end = pad_h_start + new_h
        pad_w_end = pad_w_start + new_w
        src[0, :, pad_h_start:pad_h_end, pad_w_start:pad_w_end] = img_tensor

        return src
    
    def visualize_processing(self, image: Image.Image, output_path: str = "model_processing_animation.gif"):
        image = image.convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        encoder_outputs, encoder_titles = [], []

        def add_encoder_output(tensor, title, mean_axis=0):
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()[0]
            if mean_axis is not None:
                tensor = tensor.mean(axis=mean_axis)
            encoder_outputs.append(tensor)
            encoder_titles.append(title)

        # Encoder steps
        with torch.no_grad():
            add_encoder_output(np.array(image), "Ảnh Đầu Vào\n(Kích thước: 224x224, 3 kênh RGB)", mean_axis=None)

            x = self.model.encoder.init_conv(img_tensor)
            add_encoder_output(x, "Sau Conv Ban Đầu (7x7)\n(Kích thước: 112x112, 64 kênh)")

            x = self.model.encoder.trans1(self.model.encoder.block1(x))
            add_encoder_output(x, "Sau Block1 + Trans1\n(Kích thước: 56x56, 64 kênh)")

            x = self.model.encoder.trans2(self.model.encoder.block2(x))
            add_encoder_output(x, "Sau Block2 + Trans2\n(Kích thước: 28x28, 64 kênh)")

            x = self.model.encoder.trans3(self.model.encoder.block3(x))
            add_encoder_output(x, "Sau Block3 + Trans3\n(Kích thước: 14x14, 64 kênh)")

            x = self.model.encoder.reduce_conv(x)
            add_encoder_output(x, "Sau Reduce Conv (1x1)\n(Kích thước: 14x14, 256 kênh)")

            x = self.model.encoder.encoder[0](x)
            x = self.model.encoder.encoder[1](x)
            add_encoder_output(x, "Sau Positional Encoding\n(Kích thước: 14x14, 256 kênh)")

            x = self.model.encoder.encoder[2](x)
            x_np = x.detach().cpu().numpy()[0].mean(axis=-1)
            size = int(np.sqrt(x_np.shape[0]))
            x_np = x_np.reshape(size, size) if size * size == x_np.shape[0] else x_np[None, :]
            encoder_outputs.append(x_np)
            encoder_titles.append("Đặc Trưng Phẳng\n(Kích thước: 196, 256 kênh)")

        # Decoder steps
        with torch.no_grad():
            _, probs_list, tokens_list = self.model.greedy_decode(img_tensor, self.tokenizer, max_len=10)

        decoder_probs, decoder_texts, top_indices = [], [], []
        top_k = 10
        current_tokens = []

        for step, (probs, token) in enumerate(zip(probs_list, tokens_list)):
            probs = probs[0]
            top_idxs = np.argsort(probs)[-top_k:][::-1]
            top_vals = probs[top_idxs]

            heatmap = np.zeros((step + 1, top_k))
            for i in range(step + 1):
                p = probs_list[i][0]
                heatmap[i] = p[np.argsort(p)[-top_k:][::-1]]

            current_tokens.append(token[0])
            partial_seq = self.tokenizer.decode(current_tokens)

            decoder_probs.append(heatmap)
            decoder_texts.append(f"Bước {step + 1}: {partial_seq}\n(Token ID: {token[0]})")
            top_indices.append(top_idxs)

        # Animation setup
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(1, 2, width_ratios=[20, 1])
        ax = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])

        def update(frame):
            ax.clear()
            cax.clear()

            if frame < len(encoder_outputs):
                data = encoder_outputs[frame]
                title = encoder_titles[frame]
                if data.ndim == 3:  # Ảnh RGB
                    ax.imshow(data.astype(np.uint8))
                    ax.set_title(title, fontsize=14, pad=20)
                    ax.axis('off')
                    ax.text(0.5, -0.1, "Giai đoạn Encoder: Trích xuất đặc trưng từ ảnh",
                            fontsize=12, ha='center', transform=ax.transAxes, color='blue')
                else:
                    if title.startswith("Đặc Trưng Phẳng") and data.shape[0] == 1:
                        im = ax.imshow(data, cmap='viridis', aspect='auto')
                    else:
                        im = ax.imshow(data, cmap='viridis')
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_label("Giá trị trung bình")
                    ax.set_title(title, fontsize=14, pad=20)
                    ax.axis('off')
                    ax.text(0.5, -0.1, "Giai đoạn Encoder: Trích xuất đặc trưng từ ảnh",
                            fontsize=12, ha='center', transform=ax.transAxes, color='blue')
            else:
                idx = frame - len(encoder_outputs)
                im = ax.imshow(decoder_probs[idx], cmap='viridis', aspect='auto')
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label("Xác suất")

                ax.set_title("Decoder: Dự đoán token LaTeX", fontsize=14, pad=20)
                ax.set_xlabel("Token Index (Top 10)", fontsize=12)
                ax.set_ylabel("Bước Thời Gian", fontsize=12)
                ax.set_xticks(range(top_k))
                ax.set_xticklabels([f"{i}" for i in top_indices[0]], rotation=45)

                ax.text(0.5, -0.05, f"Chuỗi LaTeX: {decoder_texts[idx]}", fontsize=10,
                    ha='center', transform=ax.transAxes, color='green',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                ax.text(0.5, -0.15, "Giai đoạn Decoder: Sinh chuỗi LaTeX từng bước",
                        fontsize=12, ha='center', transform=ax.transAxes, color='blue')

        anim = FuncAnimation(fig, update, frames=len(encoder_outputs) + len(decoder_probs), interval=1500)
        anim.save(output_path, writer=PillowWriter(fps=1))
        plt.close()
        print(f"✅ Animation đã được lưu tại: {output_path}")