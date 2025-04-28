import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.image2latex.datamodule.dataloader import ImageLatexDataManager
from src.image2latex.models.model import ImageToLatexModel

class Trainer:
    def __init__(
        self,
        data_dir: str,
        checkpoint_dir: str = "checkpoints",
        num_epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,  # L2 regularization
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
        self.ckpt_path = os.path.join(self.checkpoint_dir, "best_model_29.pt")

        self._load_data()
        self._initialize_model()
        self._resume_checkpoint()

    def _load_data(self):
        self.data_manager = ImageLatexDataManager(data_dir=self.data_dir, batch_size=self.batch_size)
        self.train_loader = self.data_manager.get_dataloader("train")
        self.val_loader = self.data_manager.get_dataloader("val")
        self.test_loader = self.data_manager.get_dataloader("test")

    def _initialize_model(self, new_lr=None):
        self.model = ImageToLatexModel(
            vocab_size=self.data_manager.vocab_size,
        )
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 là pad_id

        lr_to_use = new_lr if new_lr is not None else self.learning_rate
        self.optimizer = AdamW(self.model.parameters(), lr=lr_to_use, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=45, eta_min=1e-6)

    def _resume_checkpoint(self, reset_lr=False, new_lr=5e-6):
        if os.path.exists(self.ckpt_path):
            print("Resuming from checkpoint...")
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            if reset_lr:
                print(f"Resetting optimizer and scheduler with new learning rate: {new_lr}")
                self._initialize_model(new_lr=new_lr)
            else:
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    total_loss += loss.item()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix(train_loss=loss.item(), lr=current_lr)

                avg_train_loss = total_loss / len(self.train_loader)
                avg_val_loss = self.validate()
                self.scheduler.step()

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

    def predict_sample(self, num_samples: int = 3):
        self.model.eval()
        self.load_best_model()
        with torch.no_grad():
            for i, (src, tgt) in enumerate(self.test_loader):
                if i >= num_samples:
                    break
                src = src.to(self.device)
                preds = self.model.generate(src)
                preds_decoded = [self.data_manager.tokenizer.decode(pred.tolist()) for pred in preds]
                truths = [self.data_manager.tokenizer.decode(t.tolist()) for t in tgt]

                print("\nSample", i + 1)
                print("Pred      :", preds_decoded[0])
                # print("Pred Beam   :", preds_b[0])
                print("Truth     :", truths[0])

if __name__ == '__main__':
    trainer = Trainer(
        data_dir="data/im2latex/",
        checkpoint_dir="src/image2latex/checkpoints",
        num_epochs=45,
        batch_size=1,
        learning_rate=5e-6,
        weight_decay=1e-4
    )

    trainer._resume_checkpoint(reset_lr=True, new_lr=5e-6)
    # trainer.train()
    trainer.predict_sample(num_samples=3)