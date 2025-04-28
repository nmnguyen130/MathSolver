from PIL import Image
import numpy as np
import logging
from .latex_processing import LatexProcessor

class ImageProcessor:
    def __init__(self, crop_blank_default_size=[600, 60], pad_size=[8, 8, 8, 8], 
                 downsample_ratio=2.0, file_ext='.png', num_threads=4):
        self.crop_blank_default_size = crop_blank_default_size
        self.pad_size = pad_size  # [top, left, bottom, right]
        self.downsample_ratio = downsample_ratio
        self.file_ext = file_ext
        self.num_threads = num_threads

    def process(self, image_path):
        """Process an image: crop, pad, downsample, and save."""
        try:
            img = Image.open(image_path).convert('RGBA').convert('L')
            img_array = np.array(img)

            # Crop non-empty area
            non_empty = np.where(img_array < 255)
            if non_empty[0].size == 0:
                logging.info(f"Blank image: {image_path}, using default size {self.crop_blank_default_size}")
                cropped = Image.new('L', self.crop_blank_default_size, 255)
            else:
                y_min, y_max = non_empty[0].min(), non_empty[0].max()
                x_min, x_max = non_empty[1].min(), non_empty[1].max()
                cropped = img.crop((x_min, y_min, x_max, y_max))

            # Pad with top, left, bottom, right
            pad_top, pad_left, pad_bottom, pad_right = self.pad_size
            padded_width = cropped.width + pad_left + pad_right
            padded_height = cropped.height + pad_top + pad_bottom
            final = Image.new('L', (padded_width, padded_height), 255)
            final.paste(cropped, (pad_left, pad_top))

            # Downsample
            new_size = (int(final.width / self.downsample_ratio), int(final.height / self.downsample_ratio))
            final = final.resize(new_size, Image.LANCZOS)

            return final, (final.width, final.height)
        except Exception as e:
            logging.error(f"Failed to process image {image_path}: {str(e)}")
            return None, None, None

class DataFilter:
    def __init__(self, max_width=500, max_height=160, max_tokens=150):
        self.max_width = max_width
        self.max_height = max_height
        self.max_tokens = max_tokens
        self.latex_processor = LatexProcessor()

    def filter(self, image_path, formula, image_size):
        """Filter images and formulas based on size and token count."""
        if image_size and (image_size[0] > self.max_width or image_size[1] > self.max_height):
            logging.info(f"Discarded {image_path}: large size ({image_size[0]}x{image_size[1]})")
            return False, None

        if not formula.strip():
            logging.info(f"Discarded {image_path}: empty formula")
            return False, None

        normalized, valid = self.latex_processor.normalize_latex(formula)
        if not valid:
            logging.info(f"Discarded {image_path}: unparsable formula")
            return False, None

        if len(normalized.split()) > self.max_tokens:
            logging.info(f"Discarded {image_path}: too many tokens ({len(normalized.split())})")
            return False, None

        return True, normalized