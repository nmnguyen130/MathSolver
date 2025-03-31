import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from dataclasses import dataclass

@dataclass
class BoundingBox:
    """A bounding box for a single symbol."""
    token: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

@dataclass
class BoundingBoxes:
    """A set of bounding boxes for a complete mathematical expression."""
    label: str
    normalized_label: str | None
    bboxes: list[BoundingBox]

class BBoxLoader:
    def __init__(self, bbox_file):
        self.bbox_file = bbox_file

    def read_bbox_file(self) -> BoundingBoxes:
        """Reads a single bounding box from the input file."""
        with open(self.bbox_file, "r") as f:
            for line in f:
                bboxes = json.loads(line)

                symbol_bboxes = [
                    BoundingBox(
                        token=bbox['token'],
                        x_min=bbox['xMin'],
                        y_min=bbox['yMin'],
                        x_max=bbox['xMax'],
                        y_max=bbox['yMax'],
                    ) for bbox in bboxes['bboxes']
                ]

                return BoundingBoxes(
                    label=bboxes['label'],
                    normalized_label=bboxes.get('normalizedLabel', None),
                    bboxes=symbol_bboxes
                )
            
    def display_bboxes(self, bboxes: BoundingBoxes, figsize: tuple[int, int] = (15, 10)):
        """Displays a set of bounding boxes for debugging purposes."""
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), -float('inf'), -float('inf')
        for bbox in bboxes.bboxes:
            x_min = min(x_min, bbox.x_min)
            y_min = min(y_min, bbox.y_min)
            x_max = max(x_max, bbox.x_max)
            y_max = max(y_max, bbox.y_max)

            ax.add_patch(
                mpl_patches.Polygon(
                    ((bbox.x_min, bbox.y_min),
                     (bbox.x_min, bbox.y_max),
                     (bbox.x_max, bbox.y_max),
                     (bbox.x_max, bbox.y_min)),
                    closed=True,
                    facecolor='none',
                    edgecolor='darkblue',
                    linewidth=2))
            
        width = x_max - x_min
        height = y_max - y_min

        for bbox in bboxes.bboxes:
            box_width = bbox.x_max - bbox.x_min
            box_height = bbox.y_max - bbox.y_min
            if bbox.token != r'\frac':
                ax.text(bbox.x_min + box_width / 2,
                        bbox.y_min + box_height / 2,
                        bbox.token,
                        verticalalignment='center',
                        horizontalalignment='center',
                        fontsize=100000 / max(width, height))

        margin_ratio = 0.1
        ax.set_xlim(x_min - margin_ratio * width, x_max + margin_ratio * width)
        ax.set_ylim(y_max + margin_ratio * height, y_min - margin_ratio * height)
        plt.title(bboxes.normalized_label or bboxes.label)
        plt.savefig('BoundingBox.png')
        plt.close()