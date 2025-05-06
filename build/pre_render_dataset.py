from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def resize_image(str_image_path: str, str_output_dir: str, scale_factor: float = 0.5):
    image_path = Path(str_image_path)
    output_dir = Path(str_output_dir)
    try:
        with Image.open(image_path) as img:
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            output_path = output_dir / image_path.name
            resized_img.save(output_path, format="PNG")
            return f"Đã resize và lưu: {output_path}"
    except Exception as e:
        return f"Lỗi khi xử lý {image_path}: {e}"

def resize_images(input_dir: str, output_dir: str, scale_factor: float = 0.5, max_workers: int = 4):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob("*.png"))

    if not image_paths:
        print("Không tìm thấy ảnh nào trong thư mục.")
        return

    print(f"Tìm thấy {len(image_paths)} ảnh để xử lý.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(resize_image, str(image_path), str(output_dir), scale_factor)
            for image_path in image_paths
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Đang resize ảnh"):
            try:
                print(future.result())
            except Exception as e:
                print(f"Lỗi tiến trình: {e}")

# --- Gọi hàm ---
if __name__ == "__main__":
    pkl_file_path = "data/mathwriting-2024/processed_train_data.pkl"
    output_dir = "data/mathwriting-2024/train_image"
    resize_images(pkl_file_path, output_dir, max_workers=4)  # max_workers tùy số CPU
