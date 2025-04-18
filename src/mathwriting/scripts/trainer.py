import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.mathwriting.datamodule.dataloader import MathWritingDataManager
from src.mathwriting.models.hmre_model import MathWritingModel

class Trainer:
    def __init__(
        self,
        data_dir: str,
        checkpoint_dir: str = "checkpoints",
        num_epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,  # L2 regularization
        patience: int = 3,
        min_delta: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.device = device

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.start_epoch = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        
        self._load_data()
        self._initialize_model()
        self._resume_checkpoint()

    def _load_data(self):
        self.data_manager = MathWritingDataManager(data_dir=self.data_dir, batch_size=self.batch_size)
        self.train_loader = self.data_manager.get_dataloader("train")
        self.val_loader = self.data_manager.get_dataloader("val")

    def _initialize_model(self):
        self.model = MathWritingModel(
            vocab_size=self.data_manager.vocab_size,
            img_size=224,
            embed_dim=256
        )
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 là pad_id
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)

    def _resume_checkpoint(self):
        if os.path.exists(self.ckpt_path):
            print("Resuming from checkpoint...")
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.best_val_loss = checkpoint.get("val_loss", float("inf"))
            self.start_epoch = checkpoint.get("epoch", 0) + 1

    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss": val_loss,
        }
        filename = os.path.join(self.checkpoint_dir, f"epoch_{epoch+1:02d}.pt")
        torch.save(checkpoint, filename)
        print(f"[Checkpoint] Saved model for epoch {epoch+1} at '{filename}'")

        if is_best:
            torch.save(checkpoint, self.ckpt_path)
            print(f"[Checkpoint] Updated best model with val_loss={val_loss:.4f}")

    def summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model Summary:")
        print(f"   Device:           {self.device}")
        print(f"   Total parameters: {total_params:,}")

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
                    outputs = self.model(src, tgt_input)

                    self.optimizer.zero_grad()
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    pbar.set_postfix(train_loss=loss.item())

                avg_train_loss = total_loss / len(self.train_loader)
                avg_val_loss = self.validate()
                self.scheduler.step(avg_val_loss)

                print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                is_best = avg_val_loss < self.best_val_loss - self.min_delta
                self._save_checkpoint(epoch, avg_val_loss, is_best=is_best)

                # Early stopping
                if is_best:
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
            pbar = tqdm(self.val_loader, desc="Validating", leave=False)

            for src, tgt, _ in pbar:
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                outputs = self.model(src, tgt_input)

                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
                total_loss += loss.item()
                pbar.set_postfix(val_loss=loss.item())
        return total_loss / len(self.val_loader)
    
    def load_best_model(self):
        if os.path.exists(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            print(f"Loaded best model from {self.ckpt_path} with val_loss {checkpoint['val_loss']:.4f}")
        else:
            print("No checkpoint found!")

    def predict_sample(self, num_samples: int = 10):
        self.model.eval()
        self.load_best_model()
        with torch.no_grad():
            for i, (src, tgt, _) in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                src = src.to(self.device)
                preds_g = self.model.greedy_search(src, self.data_manager.tokenizer)
                preds_b = self.model.beam_search(src, self.data_manager.tokenizer)
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

    # trainer.train()
    trainer.predict_sample()