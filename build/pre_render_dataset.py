import pickle
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.shared.preprocessing.stroke_renderer import render_strokes_to_image

def render_single_item(args):
    index, strokes, label, image_dir = args
    try:
        img = render_strokes_to_image(strokes)
        image_path = image_dir / f"{index:06d}.png"
        img.save(image_path)
        return index, f"{index:06d}.png\t{label.strip()}\n"
    except Exception as e:
        print(f"⚠️ Error at index {index}: {e}")
        return index, None  # Skip label if error

def render_and_save_image(pkl_file_path, output_dir, max_workers=4):
    pkl_file_path = Path(pkl_file_path)
    output_dir = Path(output_dir)

    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    label_file = output_dir / "labels.txt"
    args_list = [(i, strokes, label, image_dir) for i, (strokes, label) in enumerate(data)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(render_single_item, args) for args in args_list]
        with open(label_file, "w", encoding="utf-8") as f_labels:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
                index, label_line = future.result()
                if label_line:
                    f_labels.write(label_line)

    print(f"\n✅ Render xong {len(data)} ảnh vào: {output_dir}")

# --- Gọi hàm ---
if __name__ == "__main__":
    pkl_file_path = "data/mathwriting-2024/processed_train_data.pkl"
    output_dir = "data/mathwriting-2024/train_image"
    render_and_save_image(pkl_file_path, output_dir, max_workers=4)  # max_workers tùy số CPU
