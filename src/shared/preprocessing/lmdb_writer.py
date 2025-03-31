import lmdb
import pickle
import asyncio
from tqdm import tqdm
from src.shared.preprocessing.inkml_loader import InkMLLoader

class LMDBWriter:
    def __init__(self, lmdb_path, map_size=None):
        self.env = lmdb.open(lmdb_path, map_size=map_size, subdir=False, lock=False)
    
    def write(self, data):
        with self.env.begin(write=True) as txn:
            for idx, (strokes, label) in enumerate(tqdm(data, desc="Writing LMDB")):
                txn.put(f"strokes-{idx}".encode(), pickle.dumps(strokes))
                txn.put(f"label-{idx}".encode(), label.encode())

            txn.put("length".encode(), str(len(data)).encode())  # Store dataset length

    def close(self):
        self.env.sync()
        self.env.close()

def estimate_lmdb_size(data):
    total_size = 0
    for strokes, label in data:
        total_size += len(pickle.dumps(strokes)) + len(label.encode())
    return total_size * 1.2

if __name__ == '__main__':
    train_loader = InkMLLoader('data/mathwriting-2024/train')
    train_data = asyncio.run(train_loader.load_data())

    map_size = max(16 * 1024 * 1024, estimate_lmdb_size(train_data))
    print(f"Estimated LMDB size: {map_size / 1e6:.2f} MB")

    writer = LMDBWriter('data/mathwriting-2024/train.lmdb', map_size=map_size)
    writer.write(train_data)
    writer.close()
    print("LMDB dataset saved!")