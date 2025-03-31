import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, data_dir='data/numbers', csv_path='numbers.csv', cache_file='processed_data.npz'):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, csv_path)
        self.cache_file = os.path.join(data_dir, cache_file)

    def load_data(self):
        try:
            df = pd.read_csv(self.csv_path)
            df = df[df['group'] != 'BadImag']
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

    def preprocess_image(self, image_path, target_size=(28, 28)):
        full_path = os.path.join(self.data_dir, image_path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image at {full_path}")

        original_img = img.copy()
        img = cv2.resize(img, target_size)
        
        # img = cv2.GaussianBlur(img, (3, 3), 0)
        # img = img.astype(np.float32) / 255.0

        return original_img, img
    
    def prepare_and_save_data(self, target_size=(28, 28)):
        df = self.load_data()
        images, labels = [], []
        num_samples = len(df)

        for idx, row in df.iterrows():
            try:
                _, processed_img = self.preprocess_image(row['file'], target_size)
                images.append(processed_img)
                labels.append(int(row['label']))
            except Exception as e:
                print(f"Error processing image {row['label']}: {str(e)}")

            if idx % 1000 == 0 or idx == num_samples - 1:
                print(f"Processed {idx + 1}/{num_samples} images...")

        np.savez(self.cache_file, images=np.array(images), labels=np.array(labels))
        print(f"Saved preprocessed data to {self.cache_file}")

    def visualize_samples(self, df, num_samples=6):
        indices = np.random.choice(len(df), num_samples, replace=False)
        plt.figure(figsize=(10, 6))

        for idx, sample_idx in enumerate(indices):
            original_img, processed_img = self.preprocess_image(df.iloc[sample_idx]['file'])

            plt.subplot(3, 4, idx * 2 + 1)
            plt.imshow(original_img, cmap='gray')
            plt.title("Original")
            plt.axis('off')

            plt.subplot(3, 4, idx * 2 + 2)
            plt.imshow(processed_img, cmap='gray')
            plt.title("Otsu Preprocessed")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('sample_visualization.png')
        plt.close()

if __name__ == "__main__":
    preprocess = DataPreprocessor()
    # df = preprocess.load_data()
    # preprocess.visualize_samples(df)

    preprocess.prepare_and_save_data()