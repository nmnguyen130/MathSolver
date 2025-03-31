import asyncio
import os
from src.shared.preprocessing.inkml_loader import InkMLLoader, Ink
from src.shared.preprocessing.bbox_loader import BBoxLoader
from src.shared.preprocessing.symbol_extractor import SymbolExtractor
from src.mathwriting.preprocessing.ink_renderer import InkRenderer

ROOT_DIR = "data/mathwriting-2024-excerpt"
BBOX_FILE = "synthetic-bboxes.jsonl"
SYMBOLS_FILE = "symbols.jsonl"

def main():
    bbox_path = os.path.join(ROOT_DIR, BBOX_FILE)
    symbols_path = os.path.join(ROOT_DIR, SYMBOLS_FILE)

    # Initialize dataset loaders
    train_loader = InkMLLoader(os.path.join(ROOT_DIR, "train"))
    valid_loader = InkMLLoader(os.path.join(ROOT_DIR, "valid"))
    test_loader = InkMLLoader(os.path.join(ROOT_DIR, "test"))
    bbox_loader = BBoxLoader(bbox_path)
    symbol_extractor = SymbolExtractor()

    # Load and preprocess data
    print("Loading training data...")
    train_data = asyncio.run(train_loader.load_data())
    print(f"Loaded {len(train_data)} training samples")

    print("\nLoading validation data...")
    valid_data = asyncio.run(valid_loader.load_data())
    print(f"Loaded {len(valid_data)} validation samples")

    print("\nLoading test data...")
    test_data = asyncio.run(test_loader.load_data())
    print(f"Loaded {len(test_data)} test samples")

    # Display example from training data
    if train_data:
        print("\nDisplaying first training sample...")
        example_ink = Ink(strokes=train_data[0][0], 
                         annotations={'normalizedLabel': train_data[0][1]})
        train_loader.display_ink(example_ink)
        print(f"Label: {train_data[0][1]}")

    # Load and display bounding box example if file exists
    if os.path.exists(bbox_path):
        print("\nLoading bounding box example...")
        bbox_example = bbox_loader.read_bbox_file()
        bbox_loader.display_bboxes(bbox_example)
        print(f"Bounding box label: {bbox_example.label}")
        if bbox_example.normalized_label:
            print(f"Normalized label: {bbox_example.normalized_label}")

    # Load and display symbol examples
    print("\nLoading symbols...")
    symbols = symbol_extractor.read_symbols_file(symbols_path)
    print(f"Loaded {len(symbols)} symbols")

    # Display an example symbol
    if symbols:
        print("\nDisplaying example symbol...")
        example_symbol = symbols[3902]  # Get symbol
        symbol_ink = symbol_extractor.get_symbol_ink(example_symbol)
        train_loader.display_ink(symbol_ink)
        print(f"Symbol label: {example_symbol.label}")

    # Render ink from a specific InkML file
    ink = InkMLLoader.read_inkml_file(os.path.join(ROOT_DIR, 'train', '0668dd347d600906.inkml'))
    renderer = InkRenderer()
    image = renderer.render(ink)
    image.show()

if __name__ == "__main__":
    main() 