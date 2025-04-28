import os
import tarfile
import shutil
import logging
from .data_processing import ImageProcessor, DataFilter
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

class DataExtractor:
    def __init__(self, dataset_dir, tar_file="formula_images.tar.gz"):
        self.dataset_dir = dataset_dir
        self.tar_path = os.path.join(dataset_dir, tar_file)
        self.images_dir = os.path.join(dataset_dir, "formula_images")

    def extract(self):
        """Extract tar file if images_dir doesn't exist."""
        if os.path.exists(self.images_dir):
            logging.info(f"Directory {self.images_dir} already exists, skipping extraction.")
            return
        logging.info(f"Extracting {self.tar_path}...")
        with tarfile.open(self.tar_path, 'r:gz') as tar:
            tar.extractall(self.dataset_dir, filter='data')

    def cleanup(self):
        """Remove images_dir if it exists."""
        if os.path.exists(self.images_dir):
            logging.info(f"Removing directory {self.images_dir}...")
            shutil.rmtree(self.images_dir)

class Im2LatexProcessor:
    def __init__(self, dataset_dir, output_dir, crop_blank_default_size=[600, 60], 
                 pad_size=[8, 8, 8, 8],  
                 downsample_ratio=2.0, num_threads=4, file_ext='.png', max_width=500, 
                 max_height=160, max_tokens=150):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.extractor = DataExtractor(dataset_dir)
        self.image_processor = ImageProcessor(crop_blank_default_size, pad_size, 
                                             downsample_ratio, file_ext, num_threads)
        self.filter = DataFilter(max_width, max_height, max_tokens)
        self.num_threads = num_threads
        self.file_ext = file_ext

    def process_dataset(self, lst_file, formulas_file, split_name):
        """Process dataset: load, process images, filter, and save."""

        os.makedirs(os.path.join(self.output_dir, split_name, 'images'), exist_ok=True)
        
        with open(formulas_file, 'r') as f:
            formulas = f.readlines()
        
        with open(lst_file, 'r') as f:
            lines = f.readlines()

        latex_output = os.path.join(self.output_dir, split_name, 'latex.txt')
        
        def process_item(line):
            try:
                formula_idx, image_name, _ = line.strip().split()
                formula = formulas[int(formula_idx)].strip()
                image_path = os.path.join(self.extractor.images_dir, f"{image_name}{self.file_ext}")
                output_image = os.path.join(self.output_dir, split_name, 'images', f"{image_name}{self.file_ext}")
                
                if not os.path.exists(image_path):
                    logging.info(f"Skipping {image_path}: Image not found")
                    return None
                
                # Process image
                img_result = self.image_processor.process(image_path)
                if not img_result or not all(img_result):
                    logging.info(f"Skipping {image_path}: Image processing failed")
                    return None
                
                final_img, image_size = img_result
                
                # Filter LaTeX
                valid, normalized_formula = self.filter.filter(output_image, formula, image_size)
                if not valid:
                    logging.info(f"Skipping {image_path}: LaTeX filter failed")
                    return None
                
                # Save image
                final_img.save(output_image)
                
                # Save LaTeX immediately
                with open(latex_output, "a", encoding="utf-8") as f_out:
                    f_out.write(f"{image_name}{self.file_ext}\t{normalized_formula}\n")
                
                return True
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                return None
        
        with ThreadPool(self.num_threads) as pool:
            list(tqdm(pool.imap(lambda x: process_item(x[1]), enumerate(lines)), 
                      total=len(lines), desc=f"Processing {split_name}"))
    
    def run(self, cleanup=False):
        """Run the full processing pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
            filename='logging/im2latex.log')

        self.extractor.extract()
        
        splits = [
            ("im2latex_train.lst", "train_images"),
            ("im2latex_validate.lst", "validate_images"),
            ("im2latex_test.lst", "test_images")
        ]
        
        formulas_file = os.path.join(self.dataset_dir, "im2latex_formulas.lst")
        
        for lst_file, split_name in splits:
            self.process_dataset(
                os.path.join(self.dataset_dir, lst_file),
                formulas_file,
                split_name
            )
        
        if cleanup:
            self.extractor.cleanup()

if __name__ == "__main__":
    processor = Im2LatexProcessor(
        dataset_dir="data/im2latex-100k",
        output_dir="data/processed_im2latex",
        crop_blank_default_size=[600, 60],
        pad_size=[8, 8, 8, 8],
        downsample_ratio=2.0,
        num_threads=8,
        file_ext='.png',
        max_width=500,
        max_height=160,
        max_tokens=150
    )
    processor.run(cleanup=True)