from src.digit_recognition.scripts.digit_dataset import DigitDataset
from src.digit_recognition.models.cnn import EnhancedCNN
from src.digit_recognition.scripts.trainer import ModelTrainer

def main():
    # Step1: Train model
    model = EnhancedCNN()
    trainer = ModelTrainer(
        model=model,
        batch_size=32,
        learning_rate=0.001,
        epochs=50
    )
    trainer.train()

    # Step4: Make predictions
    # predictions = trainer.predict(X)
    # print(predictions)

if __name__ == "__main__":
    main()