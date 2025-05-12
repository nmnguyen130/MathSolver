import re
from pathlib import Path
from collections import Counter

class LaTeXTokenizer:
    def __init__(self, vocab_file: Path = None):
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = {}
        self.token_to_idx = {}
        self.idx_to_token = {}
        # Regex: Xử lý lệnh LaTeX, ký tự, số, ký hiệu toán học
        # self._command_re = re.compile(
        #     r"""(
        #         \\mathbb\{[a-zA-Z]\}         |  # \mathbb{A}
        #         \\begin\{[a-z]+\}            |  # \begin{array}
        #         \\end\{[a-z]+\}              |  # \end{array}
        #         \\operatorname\*             |  # \operatorname*
        #         \\[a-zA-Z]+                  |  # \frac, \alpha, ...
        #         \\\\                         |  # double backslash
        #         \\[^a-zA-Z]                  |  # \{, \}, \%, ...
        #         [a-zA-Z0-9]                  |  # single letters or digits
        #         \S                             # any non-whitespace symbol like + - = etc
        #     )""",
        #     re.VERBOSE,
        # )
        self._command_re = re.compile(
            r"\\[a-zA-Z]+|\\.|[a-zA-Z0-9]|\S"
        )

        if vocab_file and vocab_file.exists():
            self.load_vocab(vocab_file)

        self.vocab_size = len(self.vocab)

    def build_vocab(self, latex_data: str):
        # Initialize special tokens in the vocabulary
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}

        # Count tokens in the training data
        counter = Counter()
        for text in latex_data:
            counter.update(self.tokenize(text))

        # Add tokens to the vocabulary
        for token, _ in counter.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Create token to index and index to token mappings
        self.token_to_idx = self.vocab
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

        # Save vocab to vocab.txt
        # output_file = Path("src/mathwriting/checkpoints/vocab.txt")
        # self.save_vocab(output_file)

    def tokenize(self, expression: str) -> list[str]:
        return self._command_re.findall(expression)
    
    def encode(self, expression: str) -> list[int]:
        tokens = ["<sos>"] + self.tokenize(expression) + ["<eos>"]
        unk_id = self.token_to_idx["<unk>"]
        return [self.token_to_idx.get(token, unk_id) for token in tokens]

    def decode(self, token_idx: list[int]) -> str:
        tokens = [self.idx_to_token.get(idx, "<unk>") for idx in token_idx]
        if "<eos>" in tokens:
            tokens = tokens[:tokens.index("<eos>")]
        
        decoded = []
        prev_token = ""
        for token in tokens:
            if token in self.special_tokens:
                continue
            
            # Nếu trước là lệnh LaTeX và sau là chữ/số -> thêm khoảng trắng
            if prev_token.startswith("\\") and re.match(r"[a-zA-Z0-9]", token):
                decoded.append(" ")
            
            decoded.append(token)
            prev_token = token

        return "".join(decoded)
    
    def save_vocab(self, output_file: Path):
        with open(output_file, "w", encoding="utf-8") as f:
            # Lưu mỗi token lên một dòng mới
            for token in self.vocab:
                f.write(f"{token}\n")
        print(f"Tokenizer saved to {output_file}")

    def load_vocab(self, vocab_file: Path):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]

        self.vocab = {token: idx for idx, token in enumerate(vocab)}

        self.token_to_idx = self.vocab
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
    
if __name__ == '__main__':
    tokenizer = LaTeXTokenizer()
    sample_text = r'\begin{cases}x + y > 3 \\x - y < 2\end{cases}'
    print("Tokens:", tokenizer.tokenize(sample_text))
    tokenizer.build_vocab([sample_text])
    print("Vocab:", tokenizer.vocab)
    encoded = tokenizer.encode(sample_text)
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))