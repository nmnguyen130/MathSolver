import cairo
import math
import numpy as np
import PIL
import PIL.Image

class InkRenderer:
    def __init__(self, margin=10, stroke_width=1.5, stroke_color=(0.0, 0.0, 0.0), background_color=(1.0, 1.0, 1.0)):
        self.margin = margin
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.background_color = background_color

    @staticmethod
    def cairo_to_pil(surface: cairo.ImageSurface) -> PIL.Image.Image:
        """Converts an ARGB Cairo surface into an RGB PIL image."""
        size = (surface.get_width(), surface.get_height())
        stride = surface.get_stride()
        with surface.get_data() as memory:
            return PIL.Image.frombuffer('RGB', size, memory.tobytes(), 'raw', 'BGRX', stride)
        
    def compute_canvas_size(self, ink) -> tuple[int, int, float, float]:
        """Computes the canvas size and transformation parameters."""
        xmin, ymin = np.vstack([stroke[:2].min(axis=1) for stroke in ink.strokes]).min(axis=0)
        xmax, ymax = np.vstack([stroke[:2].max(axis=1) for stroke in ink.strokes]).max(axis=0)
        width = int(xmax - xmin + 2 * self.margin)
        height = int(ymax - ymin + 2 * self.margin)
        shift_x = -xmin + self.margin
        shift_y = -ymin + self.margin
        return width, height, shift_x, shift_y
    
    def apply_transform(self, ink_x: float, ink_y: float, shift_x: float, shift_y: float) -> tuple[float, float]:
        """Applies the coordinate transformation."""
        return ink_x + shift_x, ink_y + shift_y

    def render(self, ink) -> PIL.Image.Image:
        """Renders an ink as a PIL image using Cairo."""
        width, height, shift_x, shift_y = self.compute_canvas_size(ink)
        
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(*self.background_color)
        ctx.paint()

        ctx.set_source_rgb(*self.stroke_color)
        ctx.set_line_width(self.stroke_width)
        ctx.set_line_cap(cairo.LineCap.ROUND)
        ctx.set_line_join(cairo.LineJoin.ROUND)

        for stroke in ink.strokes:
            if len(stroke[0]) == 1:
                x, y = self.apply_transform(stroke[0, 0], stroke[1, 0], shift_x, shift_y)
                ctx.arc(x, y, self.stroke_width / 2, 0, 2 * math.pi)
                ctx.fill()
            else:
                ctx.move_to(*self.apply_transform(stroke[0, 0], stroke[1, 0], shift_x, shift_y))
                for ink_x, ink_y in stroke[:2, 1:].T:
                    ctx.line_to(*self.apply_transform(ink_x, ink_y, shift_x, shift_y))
                ctx.stroke()

        return self.cairo_to_pil(surface)