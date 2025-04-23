import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.mathsolver.datamodule.dataloader import MathSolverDataManager
from src.mathsolver.models.model import MathSolverModel

class Trainer:
    def __init__(
        self,
        json_file: str,
        checkpoint_dir: str = "checkpoints",
        num_epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 3,
        min_delta: float = 0.0001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.json_file = json_file
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
        self.data_manager = MathSolverDataManager(
            json_file=self.json_file,
            batch_size=self.batch_size,
        )
        print(f"Data Manager loaded with vocab size: {self.data_manager.vocab_size}")
        self.train_loader = self.data_manager.get_dataloader("train")
        self.val_loader = self.data_manager.get_dataloader("val")

    def _initialize_model(self):
        self.model = MathSolverModel(
            vocab_size=self.data_manager.vocab_size,
            d_model=64,
            nhead=4,
            num_layers=3,
            d_ff=256,
        )
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-6)

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
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch+1}/{self.num_epochs}]")

            for i, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                graph_data = [g.to(self.device) for g in batch['graph_data']]

                target_inp = target_ids[:, :-1]  # Exclude <eos>
                target_real = target_ids[:, 1:]  # Exclude <sos>

                self.optimizer.zero_grad()
                outputs = self.model(graph_data, target_inp, input_ids)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_real.reshape(-1))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(train_loss=loss.item())

            avg_train_loss = total_loss / len(self.train_loader)
            avg_val_loss = self.validate()
            self.scheduler.step()

            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            is_best = avg_val_loss < self.best_val_loss - self.min_delta
            self._save_checkpoint(epoch, avg_val_loss, is_best=is_best)

            if is_best:
                self.best_val_loss = avg_val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            for i, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                graph_data = [g.to(self.device) for g in batch['graph_data']]

                target_inp = target_ids[:, :-1]  # Exclude <eos>
                target_real = target_ids[:, 1:]  # Exclude <sos>

                outputs = self.model(graph_data, target_inp, input_ids)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_real.reshape(-1))

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

    def predict(self, equation: str, query: str, max_len: int = 512) -> str:
        self.model.eval()
        self.load_best_model()

        # Tokenize và tạo graph từ equation
        tokens = self.data_manager.tokenizer.tokenize_latex(equation)
        tree = self.data_manager.parser.to_expr_tree(tokens)
        graph_data = self.data_manager.parser.tree_to_graph(tree, self.data_manager.tokenizer.token_to_idx)
        graph_data = [graph_data.to(self.device)]  # Đưa vào danh sách để khớp với forward
        
        # Mã hóa input (equation + query)
        input_ids = self.data_manager.tokenizer.encode_for_test(equation, query)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)  # [1, seq_len]
        
        # Khởi tạo target với <sos>
        sos_token = self.data_manager.tokenizer.token_to_idx['<sos>']
        tgt_input = torch.tensor([[sos_token]], dtype=torch.long).to(self.device)  # [1, 1]
        generated_ids = [sos_token]
       
        with torch.no_grad():
            for _ in range(max_len - 1):
                output = self.model(graph_data, tgt_input, input_tensor)

                # Get logits for the next token
                next_token_logits = output[:, -1, :]  # Shape: [1, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1)  # Shape: [1]

                # Append the predicted token
                generated_ids.append(next_token.item())
                tgt_input = torch.tensor([generated_ids], dtype=torch.long).to(self.device)  # Update tgt_input

                # Check for end-of-sequence token
                if next_token.item() == self.data_manager.tokenizer.token_to_idx['<eos>']:
                    break

        # Decode the generated token IDs
        generated = self.data_manager.tokenizer.decode(generated_ids)
        return generated if generated else "[No valid tokens generated]"

if __name__ == '__main__':
    trainer = Trainer(
        json_file="data/mathsolver/math_dataset.json",
        checkpoint_dir="src/mathsolver/checkpoints",
        num_epochs=30,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-4,
    )
    # trainer.train()
    equation = "7 + 8 ="
    query = "cộng"
    solution = trainer.predict(equation, query)
    
    print(f"Equation: {equation}")
    print(f"Query: {query}")
    print(f"Predicted Solution: {solution}")