import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.mathwriting.data.datamodule import MathWritingDataManager
from src.mathwriting.models.transformer_model import ImageToLatexModel

class Trainer:
    def __init__(
        self,
        data_dir: str,
        checkpoint_dir: str = "checkpoints",
        num_epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        patience: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.data_dir = data_dir  
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device

        self._load_data()
        self._initialize_model()

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pt")

    def _load_data(self):
        self.data_manager = MathWritingDataManager(data_dir=self.data_dir, batch_size=self.batch_size)
        self.train_loader = self.data_manager.get_dataloader("train")
        self.val_loader = self.data_manager.get_dataloader("val")

    def _initialize_model(self):
        self.model = ImageToLatexModel(vocab_size=self.data_manager.vocab_size).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

    def summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model Summary:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable:        {trainable_params:,}")

    def train(self):
        self.summary()
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch+1}/{self.num_epochs}]")

            for src, tgt, tgt_mask in pbar:
                src, tgt, tgt_mask = src.to(self.device), tgt.to(self.device), tgt_mask.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

                self.optimizer.zero_grad()
                loss = self.model.compute_loss(src, tgt_input, tgt_output, tgt_mask)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(train_loss=loss.item())

            avg_train_loss = total_loss / len(self.train_loader)
            avg_val_loss = self.validate()
            self.scheduler.step(avg_val_loss)

            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
                print("Saved new best model!")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for src, tgt, tgt_mask in self.val_loader:
                src, tgt, tgt_mask = src.to(self.device), tgt.to(self.device), tgt_mask.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

                loss = self.model.compute_loss(src, tgt_input, tgt_output, tgt_mask)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)
    
    def load_best_model(self):
        if os.path.exists(self.ckpt_path):
            self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
            print(f"Loaded best model from {self.ckpt_path}")
        else:
            print("No checkpoint found!")

    def predict_sample(self, num_samples: int = 3):
        self.model.eval()
        self.load_best_model()
        with torch.no_grad():
            for i, (src, tgt, _) in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                src = src.to(self.device)
                preds = self.model.greedy_decode(src, self.data_manager.tokenizer)
                truths = [self.data_manager.tokenizer.decode(t.tolist()) for t in tgt]
                print("\nSample", i + 1)
                print("Pred :", preds[0])
                print("Truth:", truths[0])

if __name__ == '__main__':
    trainer = Trainer(
        data_dir="data/mathwriting-2024",
        checkpoint_dir="src/mathwriting/checkpoints",
        num_epochs=30,
        batch_size=2,
        learning_rate=1e-4,
    )

    trainer.train()
    trainer.predict_sample()