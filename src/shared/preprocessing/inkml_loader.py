import os
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm_asyncio
import aiofiles
import asyncio
import pickle  # Added for caching
from xml.etree import ElementTree as ET

from dataclasses import dataclass

@dataclass
class Ink:
    """Represents a single ink, as read from an InkML file."""
    strokes: list[np.ndarray]  # Each stroke is an (x, y, timestamp) array
    annotations: dict[str, str]  # Metadata and labels extracted from the InkML file

class InkMLLoader:
    def __init__(self, folder, cache_file: str = None, num_tasks: int = 5):
        self.folder = folder
        self.cache_file = cache_file or os.path.join(folder, 'processed_ink_data.pkl')
        self.num_tasks = min(num_tasks, max(1, os.cpu_count() // 2))

    async def read_inkml_file(self, file_path) -> Ink:
        """Reads and parses a single InkML file."""
        try:
            async with aiofiles.open(file_path, "r", encoding='utf-8') as f:
                content = await f.read()
        
            root = ET.fromstring(content)
            strokes = []
            annotations = {}
            namespace = '{http://www.w3.org/2003/InkML}' # Define namespace

            for element in root.findall(f'.//{namespace}annotation'):
                annotation_type = element.attrib.get('type')
                if annotation_type and element.text:
                    annotations[annotation_type] = element.text.strip()

            for element in root.findall(f'.//{namespace}trace'):
                if element.text:
                    coords = element.text.strip().split(",")
                    stroke = [
                        (float(x), -float(y))  # Invert y-axis to match InkML's coordinate system
                        for x, y, *z in [coord.split() for coord in coords]
                    ]
                    strokes.append(np.array(stroke).T)  # Store as (3, N) array: (x, y, t)

            return Ink(strokes=strokes, annotations=annotations)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return Ink(strokes=[], annotations={})

    async def load_data(self, force_reload: bool = False):
        """Load InkML data, using cache if available."""
        # Check for cache file
        if not force_reload and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                return data
            except Exception as e:
                print(f"Error loading cache: {e}")

        # Load from InkML files if cache doesn't exist or force_reload is True
        print(f"Cache not found or reload forced. Loading from InkML files in: {self.folder}")
        if not os.path.isdir(self.folder):
            print(f"Error: Folder not found: {self.folder}")
            return []

        file_paths = [entry.path for entry in os.scandir(self.folder) if entry.is_file() and entry.name.endswith('.inkml')]
        num_files = len(file_paths)
        if num_files == 0:
            print(f"No .inkml files found in {self.folder}")
            return []

        print(f"Found {num_files} InkML files. Processing with {self.num_tasks} tasks.")
        files_per_task = (len(file_paths) + self.num_tasks - 1) // self.num_tasks

        tasks = [
            self._load_batch(file_paths[i * files_per_task: (i+1) * files_per_task])
            for i in range(self.num_tasks)
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Loading InkML files", total=len(tasks))

        data, skipped = [], 0
        for batch in results:
            for ink in batch:
                if ink.strokes:
                    # Prioritize 'normalizedLabel', fall back to 'label'
                    label = ink.annotations.get('normalizedLabel') or ink.annotations.get('label')
                    if label:
                        data.append((ink.strokes, label))
                    else:
                        skipped += 1

        print(f"Finished processing. Loaded {len(data)} samples, skipped {skipped}.")

        # Save to cache file if specified
        if self.cache_file:
            try:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(data, f)
                print("Data saved successfully to cache.")
            except Exception as e:
                print(f"Error saving cache: {e}")

        gc.collect() # Suggest garbage collection after loading large data
        return data

    async def _load_batch(self, file_paths):
        """Load a batch of InkML files."""
        batch_data = []
        for file_path in tqdm_asyncio(file_paths, desc="Processing Batch", leave=False):
            ink = await self.read_inkml_file(file_path)
            if ink:
                batch_data.append(ink)
        return batch_data

    @staticmethod
    def display_ink(ink_data, figsize=(10, 7), linewidth=2, color=None, save_path='ink_sample.png'):
        """Displays a single ink or (strokes, label) tuple."""
        if isinstance(ink_data, Ink):
            strokes = ink_data.strokes
            annotations = ink_data.annotations
            label = annotations.get('normalizedLabel') or annotations.get('label', '')
            title = f"Label: {label}"
        elif isinstance(ink_data, tuple) and len(ink_data) == 2:
            strokes, label = ink_data
            title = f"Label: {label}"
        else:
            print("Invalid input for display_ink. Provide Ink object or (strokes, label) tuple.")
            return

        plt.figure(figsize=figsize)
        for stroke in strokes:
            if stroke.shape[0] >= 2: # Ensure we have x and y
                 plt.plot(stroke[0], stroke[1], linewidth=linewidth, color=color)

        plt.title(title)
        plt.gca().set_aspect('equal', adjustable='box') # Use set_aspect for equal scaling
        plt.savefig(save_path)
        plt.close()

async def main():
    base_data_dir = os.path.join('data', 'mathwriting-2024')
    train_dir = os.path.join(base_data_dir, 'train')
    train_cache_file = os.path.join(base_data_dir, 'processed_train_data.pkl')

    train_loader = InkMLLoader(train_dir, cache_file=train_cache_file)

    start_time = time.time()
    train_data = await train_loader.load_data(force_reload=False)
    end_time = time.time()

    print(f'Training samples loaded: {len(train_data)}')
    print(f"Loading took: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    asyncio.run(main())