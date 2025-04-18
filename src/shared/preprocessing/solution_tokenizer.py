import re
from collections import Counter
from typing import Dict, List, Tuple

class SolutionTokenizer:
    def __init__(self):
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>', '<DECIMAL>', '<FRACTION>']
        self.vocab = {}
        self.token_to_idx = {}
        self.idx_to_token = {}
        self._command_re = re.compile(
            r'\\[a-zA-Z]+[0-9]*|'  # Lệnh LaTeX (như \frac, \sqrt, \log)
            r'\\[\{\}\\\|]|'  # Ký hiệu LaTeX đặc biệt (\{, \}, \|)
            r'[a-zA-ZÀ-ỹ]+|'  # Từ tiếng Việt hoặc biến (x, y, Phương, trình)
            r'\d+\.\d+|'  # Số thập phân (1.5, 0.848)
            r'\d+/\d+|'  # Phân số (21/3, 3/5)
            r'\d+|'  # Số nguyên
            r'[+\-*/=^()]|'  # Toán tử và dấu ngoặc
            r'[\.,:;?!]|'  # Dấu câu
            r'\s+|'  # Khoảng trắng
            r'[{}]|'  # Dấu ngoặc LaTeX
            r'[^\s]'  # Ký tự đơn khác
        )

    def _preprocess_number(self, text: str) -> str:
        # Thay số thập phân bằng <DECIMAL>value
        text = re.sub(r'(\d+\.\d+)', r'<DECIMAL>\g<1>', text)
        # Thay phân số bằng <FRACTION>value
        text = re.sub(r'(\d+/\d+)', r'<FRACTION>\g<1>', text)
        return text

    def build_vocab(self, train_data: List[str]):
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        counter = Counter()
        for text in train_data:
            text = self._preprocess_number(text)
            tokens = self.tokenize(text)
            counter.update(tokens)

        for token, _ in counter.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        self.token_to_idx = self.vocab
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

    def tokenize(self, expression: str) -> List[str]:
        if not expression:
            return []
        expression = self._preprocess_number(expression)
        tokens = self._command_re.findall(expression)
        # Loại bỏ khoảng trắng và trả về danh sách token
        return [token for token in tokens if not token.isspace()]
    
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def encode(self, expression: str, max_length: int = None) -> List[int]:
        tokens = ["<sos>"] + self.tokenize(expression) + ["<eos>"]
        unk_id = self.token_to_idx["<unk>"]
        encoded = [self.token_to_idx.get(token, unk_id) for token in tokens]
        if max_length:
            if len(encoded) < max_length:
                encoded += [self.token_to_idx["<pad>"]] * (max_length - len(encoded))
            else:
                encoded = encoded[:max_length]
        return encoded

    def decode(self, token_idx: List[int]) -> str:
        tokens = [self.idx_to_token.get(idx, "<unk>") for idx in token_idx]
        if "<eos>" in tokens:
            tokens = tokens[:tokens.index("<eos>")]
        return "".join(t for t in tokens if t not in self.special_tokens)
    
if __name__ == '__main__':
    # Tạo danh sách dữ liệu mẫu từ dataset
    sample_data = [
        r'Phương trình bậc hai cho trước: \\(- 2 x^{2} + 3 x - 10 = 0\\)',
        r'Phương trình: \\(\\frac{\\log{\\left(3 x \\right)}}{\\log{\\left(3 \\right)}} = 7\\)',
        r'Phương trình: \\(7 x - 6 = -6\\)',
        r'Hệ phương trình: \\(\\begin{cases} - 2 x - 3 y = 2 \\\\ - x + 2 y = 6 \\end{cases}\\)',
        r'Phương trình cho trước: \\(10 x - 3 = 3\\)',
        r'Phương trình cho trước: \\(5 \\cdot 2^{x} = 9\\)',
        r'Tích phân cho trước: \\(\\int (x^{2} - 4 x) \\, dx\\)'
    ]
    
    tokenizer = SolutionTokenizer()
    tokenizer.build_vocab(sample_data)
    
    # Kiểm tra tokenize với một mẫu
    sample_text = r'Phương trình: \\(3x + \\frac{21}{3} = -1.5\\)'
    tokens = tokenizer.tokenize(sample_text)
    print("Tokens:", tokens)
    print("Vocab size:", tokenizer.vocab)
    encoded = tokenizer.encode(sample_text)
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))