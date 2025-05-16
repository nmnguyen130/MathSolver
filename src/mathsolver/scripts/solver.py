from pathlib import Path
import torch

from src.mathsolver.models.model import MathSolverModel
from src.mathsolver.preprocessing.tokenizer import MathTokenizer

class MathSolver:
    def __init__(self, vocab_file: str, checkpoint_path: str, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        vocab_path = Path(vocab_file)
        self.tokenizer = MathTokenizer(vocab_path)

        # Load mô hình
        self.model = MathSolverModel(
            vocab_size=self.tokenizer.vocab_size,
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def solve(self, equation: str, query: str, max_len: int = 512) -> str:
        self.model.eval()

        # Mã hóa input (equation + query)
        input_ids = self.tokenizer.encode_for_test(equation, query)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)  # [1, seq_len]
        
        # Khởi tạo target với <sos>
        sos_token = self.tokenizer.token_to_idx['<sos>']
        tgt_input = torch.tensor([[sos_token]], dtype=torch.long).to(self.device)  # [1, 1]
        generated_ids = [sos_token]
       
        with torch.no_grad():
            for _ in range(max_len - 1):
                output = self.model(input_tensor, tgt_input)

                # Get logits for the next token
                next_token_logits = output[:, -1, :]  # Shape: [1, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1)  # Shape: [1]

                # Append the predicted token
                generated_ids.append(next_token.item())
                tgt_input = torch.tensor([generated_ids], dtype=torch.long).to(self.device)  # Update tgt_input

                # Check for end-of-sequence token
                if next_token.item() == self.tokenizer.token_to_idx['<eos>']:
                    break

        # Decode the generated token IDs
        generated = self.tokenizer.decode(generated_ids)
        return generated if generated else "[No valid tokens generated]"