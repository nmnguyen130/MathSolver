import re
from collections import Counter

class LatexTokenizer:
    def __init__(self):
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = {}
        self.token_to_idx = {}
        self.idx_to_token = {}
        self._command_re = re.compile(r'\\[a-zA-Z]+|\\.|[a-zA-Z0-9]|\S')

    def build_vocab(self, train_data):
        # Initialize special tokens in the vocabulary
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}

        # Count tokens in the training data
        counter = Counter()
        for text in train_data:
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
    
if __name__ == '__main__':
    tokenizer = LatexTokenizer()
    sample_text = r'\vartheta(n)=\frac{1}{\sqrt{1-\frac{n^{7}}{c^{7}}}}'
    print("Tokens:", tokenizer.tokenize(sample_text))
    tokenizer.build_vocab([sample_text])
    print("Vocab:", tokenizer.vocab)
    encoded = tokenizer.encode(sample_text)
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))