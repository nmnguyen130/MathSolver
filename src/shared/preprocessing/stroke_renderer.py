import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configure matplotlib to use a non-interactive backend to avoid display windows
plt.switch_backend('Agg') 

def render_strokes_to_image(strokes):
    """
    Renders a list of strokes into a PIL Image.

    Args:
        strokes: A list of strokes, where each stroke is a list of (x, y) coordinates.

    Returns:
        A PIL Image object representing the rendered strokes.
    """
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_aspect('equal')

    for stroke_data in strokes:
        # Ensure stroke_data is a valid array/list format
        if isinstance(stroke_data, (np.ndarray, list)):
            coords = stroke_data.tolist() if isinstance(stroke_data, np.ndarray) else stroke_data
            
            # Check for shape (N, 2) - N points, (x, y)
            if len(coords) == 2:
                x, y = coords
                ax.plot(x, y, color='black', linewidth=2)
            else:
                print(f"Warning: Skipping stroke with invalid coordinate format: {coords}")
                continue

    # Save the plot to a buffer and return as a PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)

    img = Image.open(buf).convert('RGB')
    return img