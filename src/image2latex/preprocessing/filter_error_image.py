import os
import gc
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from multiprocessing import Pool

folder_path = 'data/im2latex/test/images'
latex_file = 'data/im2latex/test/latex.txt'

def check_image(filename):
    image_path = os.path.join(folder_path, filename)
    if os.path.isfile(image_path):
        try:
            with Image.open(image_path) as img:
                img.convert("RGB")
                img.close()
            return None
        except (UnidentifiedImageError, OSError):
            return filename
    return None

def clean_corrupted_images(corrupted_images, folder_path, latex_file):
    # Xóa file ảnh lỗi
    for filename in corrupted_images:
        image_path = os.path.join(folder_path, filename)
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"Đã xóa ảnh lỗi: {filename}")
            except Exception as e:
                print(f"Lỗi khi xóa {filename}: {e}")

    # Đọc toàn bộ latex file
    with open(latex_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Lọc bỏ các dòng chứa ảnh lỗi
    new_lines = []
    corrupted_set = set(corrupted_images)  # Tăng tốc lookup
    for line in lines:
        image_name = line.split('\t')[0]
        if image_name not in corrupted_set:
            new_lines.append(line)

    # Ghi lại file latex.txt đã clean
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"\nĐã xóa {len(corrupted_images)} ảnh và cập nhật lại file latex.txt.")

if __name__ == "__main__":
    all_files = os.listdir(folder_path)
    corrupted_images = []

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(check_image, all_files), total=len(all_files), desc="Đang kiểm tra ảnh"))

    corrupted_images = [res for res in results if res is not None]
    gc.collect()

    print("\nDanh sách ảnh lỗi:")
    print(corrupted_images)
    print(f"\nTổng cộng {len(corrupted_images)} ảnh bị lỗi.")

    # Gọi hàm clean
    clean_corrupted_images(corrupted_images, folder_path, latex_file)
