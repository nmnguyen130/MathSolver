import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import transforms
from PIL import Image

# Tạo dữ liệu giả cho animation
def create_input_image():
    """Tạo ảnh đầu vào giả (ảnh đen trắng đơn giản)."""
    return np.random.rand(64, 64)  # Mô phỏng ảnh 64x64

def generate_feature_map(input_image):
    """Giả lập việc tạo feature map từ Encoder."""
    return np.random.rand(64, 64)  # Chuyển thành một feature map giả

def apply_positional_encoding(feature_map):
    """Giả lập việc áp dụng Positional Encoding."""
    return feature_map + np.sin(np.linspace(0, 2 * np.pi, feature_map.shape[0]))  # Một biến đổi đơn giản

def transformer_decoder(pos_encoded_features):
    """Giả lập bước Decoder của Transformer."""
    return np.random.rand(64, 64)  # Mô phỏng kết quả cuối cùng

def update_frame(frame_num, input_img, feature_map_img, pos_encoded_img, decoder_img):
    """Cập nhật các bước cho animation."""
    if frame_num == 0:
        input_img.set_array(create_input_image())  # Bước 0: Ảnh đầu vào
    elif frame_num == 1:
        feature_map_img.set_array(generate_feature_map(input_img.get_array()))  # Bước 1: Encoder
    elif frame_num == 2:
        pos_encoded_img.set_array(apply_positional_encoding(feature_map_img.get_array()))  # Bước 2: Positional Encoding
    elif frame_num == 3:
        decoder_img.set_array(transformer_decoder(pos_encoded_img.get_array()))  # Bước 3: Decoder

    return input_img, feature_map_img, pos_encoded_img, decoder_img

# Tạo figure cho animation
fig, axs = plt.subplots(1, 4, figsize=(12, 4))

# Tạo ảnh trống cho từng bước
input_img = axs[0].imshow(np.zeros((64, 64)), cmap='gray')
axs[0].set_title("Input Image")

feature_map_img = axs[1].imshow(np.zeros((64, 64)), cmap='hot')
axs[1].set_title("Feature Map")

pos_encoded_img = axs[2].imshow(np.zeros((64, 64)), cmap='coolwarm')
axs[2].set_title("Positional Encoding")

decoder_img = axs[3].imshow(np.zeros((64, 64)), cmap='viridis')
axs[3].set_title("Decoder Output")

# Tạo animation
ani = animation.FuncAnimation(
    fig, update_frame, frames=4, interval=1000, blit=False
)

# Hiển thị animation
plt.show()