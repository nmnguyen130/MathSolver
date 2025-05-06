from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def resize_image(image_path: Path, output_dir: Path, scale_factor: float = 0.5):
    """
    Resize một ảnh và lưu vào thư mục output.
    
    Args:
        image_path (Path): Đường dẫn tới ảnh gốc.
        output_dir (Path): Thư mục lưu ảnh đã resize.
        scale_factor (float): Tỷ lệ giảm kích thước (mặc định 0.5 = nhỏ hơn 2 lần).
    
    Returns:
        str: Thông báo kết quả.
    """
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
    """
    Resize tất cả ảnh trong input_dir nhỏ hơn 2 lần và lưu vào output_dir, sử dụng đa tiến trình.
    
    Args:
        input_dir (str): Đường dẫn thư mục chứa ảnh gốc.
        output_dir (str): Đường dẫn thư mục để lưu ảnh đã resize.
        scale_factor (float): Tỷ lệ giảm kích thước (mặc định 0.5 = nhỏ hơn 2 lần).
        max_workers (int): Số lượng tiến trình tối đa (mặc định 4).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Tạo thư mục output nếu chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách file ảnh
    image_paths = list(input_dir.glob("*.png"))  # Chỉ xử lý file PNG, có thể thay đổi nếu cần
    
    if not image_paths:
        print("Không tìm thấy ảnh nào trong thư mục.")
        return
    
    print(f"Tìm thấy {len(image_paths)} ảnh để xử lý.")
    
    # Sử dụng ProcessPoolExecutor để xử lý song song
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Tạo danh sách công việc
        futures = [
            executor.submit(resize_image, image_path, output_dir, scale_factor)
            for image_path in image_paths
        ]
        
        # Hiển thị tiến độ với tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Đang resize ảnh"):
            print(future.result())

# Sử dụng hàm
if __name__ == "__main__":
    input_dir = "data/mathwriting-2024/valid_image/images"
    output_dir = "data/mathwriting-2024/valid_image/resized_images"
    resize_images(input_dir, output_dir, scale_factor=0.5, max_workers=4)