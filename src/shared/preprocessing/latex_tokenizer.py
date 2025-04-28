import re
from collections import Counter

class LaTeXTokenizer:
    def __init__(self):
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = {}
        self.token_to_idx = {}
        self.idx_to_token = {}
        # Regex: Xử lý lệnh LaTeX, ký tự, số, ký hiệu toán học
        self._command_re = re.compile(
            r'('
            r'\\(?:begin|end)\{[a-zA-Z]+\}|'  # \begin{matrix}, \end{matrix}
            r'\\[a-zA-Z]+(?:{[a-zA-Z]*})?|'  # \frac, \mathbb{A}, \begin{cases}
            r'\\[\[\]{}()|]|'                # \{, \}, \[, \], \(, \), \|
            r'[0-9]+|'                       # 0-9, 123
            r'[a-zA-Z]|'                     # a-z, A-Z
            r'[\,\;\:\!\?\.]|'               # , ; : ! ? .
            r'[\{\}\[\]\(\)]|'               # { } [ ] ( )
            r'[\*\/+\-\_=><\^~]|'            # * / + - _ = > < ^ ~
            r'[&\#\%\|]|'                    # & # % |
            r'\\|'                           # Dấu \ riêng
            r'\s+'                           # Khoảng trắng
            r')'
        )

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
        return "".join(token for token in tokens if token not in self.special_tokens)
    
    def save_vocab(self, output_file: str) -> None:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('{\n')
            for token, idx in self.vocab.items():
                f.write(f'  "{token}": {idx},\n')
            f.write('}\n')
        print(f"Tokenizer saved to {output_file}")

    def load_vocab(self, vocab_file: str) -> None:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = eval(f.read())  # Giả sử file JSON đơn giản
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