import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.mathwriting.dataloader.datamodule import MathWritingDataManager
from src.mathwriting.models.transformer_model import ImageToLatexModel

class Trainer:
    def __init__(
        self,
        data_dir: str,
        checkpoint_dir: str = "checkpoints",
        num_epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,  # L2 regularization
        patience: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device

        self._load_data()
        self._initialize_model()

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.start_epoch = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        
        self._resume_checkpoint()

    def _load_data(self):
        self.data_manager = MathWritingDataManager(data_dir=self.data_dir, batch_size=self.batch_size)
        self.train_loader = self.data_manager.get_dataloader("train")
        self.val_loader = self.data_manager.get_dataloader("val")

    def _initialize_model(self):
        self.model = ImageToLatexModel(
            vocab_size=self.data_manager.vocab_size,
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            num_layers=3
        )
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)

    def _resume_checkpoint(self):
        if os.path.exists(self.ckpt_path):
            print("Resuming from checkpoint...")
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.start_epoch = checkpoint.get("epoch", 0) + 1

    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        filename = os.path.join(self.checkpoint_dir, f"epoch_{epoch+1:02d}.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, filename)
        print(f"[Checkpoint] Saved model for epoch {epoch+1} at '{filename}'")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] Updated best model with val_loss={val_loss:.4f}")

    def summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model Summary:")
        print(f"   Device:           {self.device}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable:        {trainable_params:,}")

    def train(self):
        self.summary()
        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                self.model.train()
                total_loss = 0.0
                pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch+1}/{self.num_epochs}]")

                for src, tgt, _ in pbar:
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]
                    tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1))

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
                self._save_checkpoint(epoch, avg_val_loss, is_best=(avg_val_loss < self.best_val_loss))

                # Early stopping
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.patience:
                        print("Early stopping triggered.")
                        break
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current model...")
            self._save_checkpoint(epoch, avg_val_loss, is_best=False)
            print("Checkpoint saved. Exiting training.")

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for src, tgt, _ in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1))

                loss = self.model.compute_loss(src, tgt_input, tgt_output, tgt_mask)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)
    
    def load_best_model(self):
        if os.path.exists(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
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
                preds_g = self.model.greedy_decode(src, self.data_manager.tokenizer)
                preds_b = self.model.beam_decode(src, self.data_manager.tokenizer)
                truths = [self.data_manager.tokenizer.decode(t.tolist()) for t in tgt]
                print("\nSample", i + 1)
                print("Pred Greedy :", preds_g[0])
                print("Pred Beam   :", preds_b[0])
                print("Truth       :", truths[0])

if __name__ == '__main__':
    trainer = Trainer(
        data_dir="data/mathwriting-2024",
        checkpoint_dir="src/mathwriting/checkpoints",
        num_epochs=30,
        batch_size=16,
        learning_rate=1e-4,
    )

    trainer.train()
    trainer.predict_sample()