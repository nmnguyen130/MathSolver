import os
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm_asyncio
import aiofiles
import asyncio
from xml.etree import ElementTree as ET

from dataclasses import dataclass

@dataclass
class Ink:
    """Represents a single ink, as read from an InkML file."""
    strokes: list[np.ndarray]  # Each stroke is an (x, y, timestamp) array
    annotations: dict[str, str]  # Metadata and labels extracted from the InkML file

class InkMLLoader:
    def __init__(self, folder, num_tasks: int = 10):
        self.folder = folder
        self.num_tasks = min(num_tasks, max(1, os.cpu_count() // 2))

    async def read_inkml_file(self, file_path) -> Ink:
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()

        root = ET.fromstring(content)
        strokes = []
        annotations = {}

        for element in root.iter('{http://www.w3.org/2003/InkML}annotation'):
            # Process annotation elements
            annotations[element.attrib.get('type')] = element.text

        for element in root.iter('{http://www.w3.org/2003/InkML}trace'):
            # Process trace elements
            try:
                points = np.fromstring(element.text.replace(',', ' '), sep=' ').reshape(-1, 3)
                strokes.append(points.T)    # (x, y, t)
            except ValueError:
                continue

        return Ink(strokes=strokes, annotations=annotations)

    async def load_data(self):
        """Load all InkML files from the folder."""
        file_paths = [entry.path for entry in os.scandir(self.folder) if entry.name.endswith('.inkml')]
        num_files = len(file_paths)
        files_per_task = max(1, (num_files + self.num_tasks - 1) // self.num_tasks)

        tasks = [
            self._load_batch(file_paths[i * files_per_task: (i+1) * files_per_task])
            for i in range(self.num_tasks)
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Loading InkML files", total=len(tasks))
        data = []
        
        for batch in results:
            for ink in batch:
                if ink.strokes:
                    label = ink.annotations.get('normalizedLabel') or ink.annotations.get('label')
                    if label:
                        data.append((ink.strokes, label))
        
        return data
    
    async def _load_batch(self, file_paths):
        batch_data = []
        for file_path in tqdm_asyncio(file_paths, desc="Processing Batch", leave=False):
            ink = await self.read_inkml_file(file_path)
            batch_data.append(ink)

        return batch_data

    @staticmethod
    def display_ink(ink: Ink, figsize: tuple[int, int] = (15, 10), linewidth: int = 2, color=None):
        """Simple display for a single ink."""
        plt.figure(figsize=figsize)
        for stroke in ink.strokes:
            plt.plot(stroke[0], stroke[1], linewidth=linewidth, color=color)
        plt.title(
            f"{ink.annotations.get('sampleId', '')} -- "
            f"{ink.annotations.get('splitTagOriginal', '')} -- "
            f"{ink.annotations.get('normalizedLabel', ink.annotations.get('label', ''))}"
        )  # Set the title with annotations
        plt.gca().invert_yaxis()  # Invert the y-axis for correct orientation
        plt.gca().axis('equal')  # Set equal scaling for both axes
        plt.savefig('ink_sample.png')
        plt.close()

async def main():
    train_loader = InkMLLoader('data/mathwriting-2024/train')
    train_data = await train_loader.load_data()
    print(f'Training samples: {len(train_data)}')
    
    train_ink = Ink(strokes=train_data[0][0], annotations={'normalizedLabel': train_data[0][1]})
    train_loader.display_ink(train_ink)

if __name__ == '__main__':
    asyncio.run(main())