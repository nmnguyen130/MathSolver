from pathlib import Path
import json

class LaTeXTokenizer:
    def __init__(self, vocab_file: Path = None):
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.vocab = {}
        self.token_to_idx = {}
        self.idx_to_token = {}

        if vocab_file and vocab_file.exists():
            self.load_vocab(vocab_file)

    def save_vocab(self, output_file: Path):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {output_file}")

    def load_vocab(self, vocab_file: Path):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]

        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        for token in vocab:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.token_to_idx = self.vocab
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

    def build_vocab(self, train_labels: list[str]):
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        for label in train_labels:
            tokens = label.split(" ")
            for token in tokens:
                if token and token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        self.token_to_idx = self.vocab
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Save vocab to vocab.txt
        # output_file = Path("vocab.txt")
        # self.save_vocab(output_file)

    def tokenize(self, expression: str) -> list[str]:
        return expression.strip().split()

    def encode(self, expression: list[str]) -> list[int]:
        tokens = ["<sos>"] + self.tokenize(expression) + ["<eos>"]
        unk_id = self.token_to_idx["<unk>"]
        return [self.token_to_idx.get(token, unk_id) for token in tokens]

    def decode(self, token_idx: list[int]) -> str:
        tokens = [self.idx_to_token.get(idx, "<unk>") for idx in token_idx]
        if "<eos>" in tokens:
            tokens = tokens[:tokens.index("<eos>")]
        return " ".join(token for token in tokens if token not in self.special_tokens)