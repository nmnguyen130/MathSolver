import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

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

def normalize_strokes(strokes, image_size):
    all_x = np.concatenate([s[0] for s in strokes])
    all_y = np.concatenate([s[1] for s in strokes])

    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()

    width, height = image_size
    scale_x = (width - 1) / (max_x - min_x + 1e-6)
    scale_y = (height - 1) / (max_y - min_y + 1e-6)
    scale = min(scale_x, scale_y)

    drawing_width = (max_x - min_x) * scale
    drawing_height = (max_y - min_y) * scale

    # Offset để căn giữa
    offset_x = (width - drawing_width) / 2
    offset_y = (height - drawing_height) / 2

    norm_strokes = []
    for x, y in strokes:
        x_norm = (np.array(x) - min_x) * scale + offset_x
        y_norm = (max_y - np.array(y)) * scale + offset_y  # Lật trục y và offset
        norm_strokes.append([x_norm.tolist(), y_norm.tolist()])

    return norm_strokes

def render_strokes_to_image_fast(strokes, image_size=(256, 256)):
    # Create a blank white image
    img = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(img)

    norm_strokes = normalize_strokes(strokes, image_size)

    for stroke_data in norm_strokes:
        x, y = stroke_data
        if len(x) >= 2:
            points = list(zip(x, y))
            draw.line(points, fill="black", width=2)

    return img