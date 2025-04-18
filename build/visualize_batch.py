import time
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.mathwriting.datamodule.dataloader import MathWritingDataManager

DATA_DIR = "./data/mathwriting-2024/"
BATCH_SIZE = 16 # Number of samples to visualize
NUM_WORKERS = 0 # Keep this 0 for visualization script unless needed

def visualize_batch():
    """Loads a batch of data and visualizes images and labels."""
    print(f"Loading data from: {DATA_DIR}")

    if not Path(DATA_DIR).exists():
        print(f"Error: Data directory not found at {DATA_DIR}")
        return

    try:
        # 1. Instantiate the Data Manager
        print("Instantiating DataManager...")
        data_manager = MathWritingDataManager(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )

        # 2. Get the tokenizer
        tokenizer = data_manager.tokenizer
        if tokenizer is None:
            print("Error: Tokenizer was not initialized by DataManager.")
            return
        print(f"Tokenizer loaded with vocab size: {data_manager.vocab_size}")

        # 3. Get the training dataloader
        print("Getting train dataloader...")
        train_dataloader = data_manager.get_dataloader('train')
        if train_dataloader is None:
            print("Error: Failed to get train dataloader.")
            return

        # 4. Fetch one batch
        print(f"Fetching one batch of size {BATCH_SIZE}...")
        try:
            batch = next(iter(train_dataloader))
            src, tgt, tgt_mask = batch # Assuming collate_fn returns these three
        except StopIteration:
            print("Error: Could not fetch a batch from the dataloader (dataloader might be empty).")
            return
        except Exception as e:
            print(f"Error fetching batch: {e}")
            return

        print(f"Batch loaded. Image tensor shape: {src.shape}, Label tensor shape: {tgt.shape}")
        print(f"Image tensor min: {src.min()}, max: {src.max()}")  # Thêm kiểm tra giá trị

        # 5. Visualize Images
        print("\nVisualizing images...")
        plt.figure(figsize=(16, 16))
        try:
            grid_img = make_grid(src, nrow=int(BATCH_SIZE**0.5))
            # Check if images are normalized (0-1) or not (0-255) for display
            if grid_img.max() <= 1.0 and grid_img.min() >= 0.0:
                plt.imshow(grid_img.permute(1, 2, 0))
            else:
                # Assuming ToTensor() was used, which scales to [0, 1]
                # If not, might need different handling or normalization check
                plt.imshow(grid_img.permute(1, 2, 0))
                print("Warning: Image tensor values outside [0, 1]. Display might be incorrect if not scaled.")

            plt.title(f"Batch of {BATCH_SIZE} Handwritten Math Expressions")
            plt.axis("off")
            plt.savefig("visualized_batch.png")
            plt.close()
        except Exception as e:
            print(f"Error displaying images: {e}")

        # 6. Decode and Print Labels
        print("\nDecoded LaTeX Labels for the batch:")
        pad_id = getattr(tokenizer, 'PAD_ID', 0) # Get pad id from tokenizer or default to 0
        for i, t in enumerate(tgt):
            # Filter out padding tokens before decoding
            label_tokens = [token.item() for token in t if token.item() != pad_id]
            # Decode, handling potential SOS/EOS if tokenizer adds them and decode doesn't remove them
            decoded_label = tokenizer.decode(label_tokens)
            print(f"  Sample {i}: {decoded_label}")

    except Exception as e:
        print(f"Unexpected error: {e}")

def benchmark_loading():
    print(f"Benchmarking data loading from: {DATA_DIR}")
    if not Path(DATA_DIR).exists():
        print(f"Error: Data directory not found at {DATA_DIR}")
        return

    data_manager = MathWritingDataManager(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    train_dataloader = data_manager.get_dataloader("train")

    print("Starting benchmark over full dataset...")
    total_batches = len(train_dataloader)

    print(f"Batch size: {BATCH_SIZE} | Total batches: {total_batches}")
    
    start_time = time.time()
    for i, batch in enumerate(train_dataloader):
        src, tgt, tgt_mask = batch
        if i % 10 == 0:
            print(f"Processed {i}/{total_batches} batches...")
        if i % 100 == 0 and i != 0:
            break
    end_time = time.time()

    duration = end_time - start_time
    print(f"\n✅ Completed loading entire dataset in {duration:.2f} seconds")
    print(f"⏱ Avg time per batch: {duration / total_batches:.3f} seconds")

if __name__ == "__main__":
    visualize_batch()
