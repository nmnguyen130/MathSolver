import cv2
import torch
import numpy as np
from PIL import ImageGrab
from src.digit_recognition.models.cnn import EnhancedCNN

class DigitDetector():
    def __init__(self, model_path='best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EnhancedCNN().to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.eval()

    def detect_digit(self, input_tensor, threshold=0.7):
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = prob.max(1)

            if confidence.item() < threshold:
                return -1
            return predicted.item()  # Return the predicted digit
        
    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        return thresh
    
    def extract_digits(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digits = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 and h > 10:
                digit = img[y:y+h, x:x+w]
                digit = cv2.resize(digit, (28, 28))
                digit = digit / 255.0
                digit = torch.tensor(digit, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                digits.append((digit, (x, y, w, h)))
        return digits
    
    def run(self, image_path='Capture.PNG'):
        screenshot = cv2.imread(image_path)
        processed = self.preprocess_image(screenshot)
        digits = self.extract_digits(processed)

        for digit, (x, y, w, h) in digits:
            number = self.detect_digit(digit)
            if number != -1:
                cv2.rectangle(screenshot, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(screenshot, str(number), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Detected Digits', screenshot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DigitDetector()
    detector.run('Capture.PNG')
