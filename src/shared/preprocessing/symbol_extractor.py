import os
import json
from dataclasses import dataclass
from src.shared.preprocessing.inkml_loader import Ink, InkMLLoader

MATHWRITING_ROOT_DIR = "data/mathwriting-2024-excerpt"

@dataclass
class InkPart:
    """A pointer to a part of an ink corresponding to a single symbol."""
    source_sample_id: str
    label: str
    stroke_indices: list[int]

class SymbolExtractor:
    def read_symbols_file(self, symbols_file) -> list[InkPart]:
        symbols = []
        with open(symbols_file, "r") as f:
            for line in f:
                symbol_json = json.loads(line)
                symbols.append(InkPart(
                    source_sample_id=symbol_json['sourceSampleId'],
                    label=symbol_json['label'],
                    stroke_indices=symbol_json['strokeIndices']
                ))
        return symbols
    
    def get_symbol_ink(self, symbol: InkPart) -> Ink:
        """Computes the actual ink from an InkPart object."""
        ink = InkMLLoader.read_inkml_file(
            os.path.join(MATHWRITING_ROOT_DIR, "train", f"{symbol.source_sample_id}.inkml")
        )
        strokes = [ink.strokes[i] for i in symbol.stroke_indices]
        return Ink(
            strokes=strokes,
            annotations={
                'label': symbol.label,
                'splitTagOriginal': ink.annotations['splitTagOriginal']
            }
        )