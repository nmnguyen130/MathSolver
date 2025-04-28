from pathlib import Path

def generate_latex_files(data_dir: str):
    data_dir = Path(data_dir)
    latex_file = data_dir / "latex.txt"
    splits = ["train", "valid", "test"]

    # Đọc toàn bộ nhãn LaTeX
    with open(latex_file, "r", encoding="utf-8") as f:
        latex_labels = [line.strip() for line in f]

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Thư mục {split} không tồn tại, bỏ qua.")
            continue

        # Lấy danh sách ảnh
        image_files = sorted(split_dir.glob("images/*.png"))
        split_labels = []

        missing_latex_images = []
        empty_label_images = []

        for img_path in image_files:
            try:
                img_index = int(img_path.stem)
            except ValueError:
                print(f"Cảnh báo: Tên file {img_path.name} không hợp lệ, bỏ qua.")
                continue

            if img_index < len(latex_labels):
                label = latex_labels[img_index].strip()
                if not label:
                    print(f"Warning: Label rỗng cho ảnh {img_path.name}")
                    empty_label_images.append(img_path.name)
                    continue  # bỏ luôn ảnh này
                # Nếu label ổn thì nối
                split_labels.append(f"{img_path.name}\t{label}")
            else:
                print(f"Warning: Ảnh {img_path.name} không có nhãn LaTeX tương ứng.")
                missing_latex_images.append(img_path.name)

        if split == "train":
            custom_image_name = "0234884.png"
            custom_label = r"\left( x - 1 \right) ^ { 2 } + \left( 1 - x ^ { 2 } \right) = 1"
            split_labels.append(f"{custom_image_name}\t{custom_label}")
            print(f"Đã thêm thủ công ảnh {custom_image_name} vào train/latex.txt")

        # Lưu file latex.txt mới
        output_file = split_dir / "latex.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(split_labels))
        print(f"Đã tạo {output_file} với {len(split_labels)} dòng hợp lệ.")

        # Báo cáo các ảnh lỗi
        if missing_latex_images:
            print(f"\nẢnh trong {split}/images không có nhãn:")
            for img_name in missing_latex_images:
                print(f"  - {img_name}")

        if empty_label_images:
            print(f"\nẢnh trong {split}/images có label rỗng:")
            for img_name in empty_label_images:
                print(f"  - {img_name}")

if __name__ == "__main__":
    generate_latex_files("./data/im2latex")
