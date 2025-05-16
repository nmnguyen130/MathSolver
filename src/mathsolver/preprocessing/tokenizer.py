import json
import os
from pathlib import Path
import re
from collections import Counter
from typing import List, Tuple

class MathTokenizer:
    def __init__(self, vocab_file: Path = None):
        # Special tokens
        self.special_tokens = [
            '<pad>', '<sos>', '<eos>', '<unk>',
            '<NUM>', '</NUM>',
            '<EQ>', '</EQ>', '<QUERY>', '</QUERY>', '<STEP>', '</STEP>', '<ANSWER>', '</ANSWER>',
            '\\(', '\\)',
        ]
        self.operators = ['+', '-', '\\times', '\\cdot', '\\div', '=', '^', '\\pm',
                          '<', '>', '\\leq', '\\geq']
        self.brackets = ['(', ')', '{', '}', '\\left', '\\right', '[', ']']
        self.latex_commands = ['\\frac', '\\sqrt', '\\log', '\\left', '\\right', '\\Delta', 
                               '\\wedge', '\\infty', '\\begin', '\\end', '\\\\', 'cases']
        self.math_functions = ['\\sin', '\\cos', '\\tan', '\\log', '\\exp', '\\ln']

        self.vocab = {}
        self.token_to_idx = {}
        self.idx_to_token = {}
        # Regex cho LaTeX: tách chi tiết từng thành phần        
        self._latex_re = re.compile(
            r"[a-zA-Z]+'+|"              # Từ f', f'', g'
            r'\\[a-zA-Z]+|'              # Lệnh LaTeX (như \sqrt, \frac)
            r'\\{|\\}|'                  # Mở, đóng nhóm LaTeX đặc biệt
            r'\{|\}|'                    # Mở, đóng nhóm
            r'[0-9]|\.|'                 # Từng chữ số, dấu chấm
            r'hoặc|và|là|'
            r'[a-zA-Z]+(?:\^\{.*?\})?|'  # Biến (như x, x^{2})
            r'[\+\-\*\/=^()]|'           # Toán tử
            r'\\[\+\-\*\/=^\\]|'         # Toán tử LaTeX (như \\pm)
            r'\S'                        # Ký tự khác
        )
        # Regex cho tiếng Việt: tách từ, ký tự đặc biệt, chữ số riêng lẻ
        self._vietnamese_re = re.compile(
            r'[^\s\d+\-*/=^()\\]+|'      # Từ tiếng việt
            r'[\+\-\*\/=^()]|'           # Toán tử
            r'[0-9]|\.|'                 # Từng chữ số, dấu chấm
            r'\S'                        # Ký tự khác
        )

        if vocab_file and vocab_file.exists():
            self.load_vocab(vocab_file)

        self.vocab_size = len(self.vocab)

    def _preprocess_common(self, text: str) -> str:
        """Tiền xử lý chung: tách dấu câu và ngoặc, chuẩn hóa khoảng trắng."""
        text = re.sub(r'([,.!?:])', r' \1 ', text)
        text = re.sub(r'([{}()])', r' \1 ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess_latex(self, expression: str) -> str:
        """Tiền xử lý biểu thức LaTeX để tách nhóm phức tạp."""
        # Thay thế các nhóm {content} để đảm bảo tách đúng
        expression = re.sub(r'\{([^\{\}]*?)\}', r' { \1 } ', expression)
        # Thêm khoảng trắng quanh toán tử và lệnh LaTeX
        expression = re.sub(r'(\\[a-zA-Z]+)', r' \1 ', expression)
        expression = re.sub(r'([\+\-\*\/=^()])', r' \1 ', expression)
        return self._preprocess_common(expression)
    
    def preprocess_text(self, text: str) -> str:
        return self._preprocess_common(text.lower())
    
    def _is_number_start(self, token: str, next_token: str, last_token: str) -> bool:
        """Kiểm tra xem token có phải là khởi đầu của số (bao gồm số âm/dương)."""
        if token not in ['-', '+'] or not next_token or not next_token.isdigit():
            return False

        if last_token is None or last_token in self.operators or last_token in self.brackets:
            return True

        if last_token not in self.special_tokens:
            return True

        return False
    
    def _tokenize_number(self, tokens: List[str], i: int, last_token: str) -> Tuple[List[str], int, str]:
        """Xử lý số (bao gồm số âm/dương và thập phân)."""
        num_tokens = [tokens[i]]
        i += 1
        while i < len(tokens) and (tokens[i].isdigit() or tokens[i] == '.'):
            num_tokens.append(tokens[i])
            i += 1
        return ['<NUM>'] + num_tokens + ['</NUM>'], i, '<NUM>'
    
    def _tokenize_combined(self, token: str) -> List[str]:
        """Xử lý trường hợp số kèm biến (như 4x)."""
        return ['<NUM>', token[:-1], '</NUM>', token[-1]]
    
    def _tokenize_common(self, tokens: List[str], regex: re.Pattern) -> List[str]:
        """Tách token chung cho cả LaTeX và tiếng Việt."""
        result = []
        last_token = None
        i = 0
        while i < len(tokens):
            token = tokens[i]
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None

            if self._is_number_start(token, next_token, last_token):
                num_tokens, i, last_token = self._tokenize_number(tokens, i, last_token)
                result += num_tokens
            elif token.isdigit() or (token == '.' and next_token and next_token.isdigit()):
                num_tokens, i, last_token = self._tokenize_number(tokens, i, last_token)
                result += num_tokens
            elif re.fullmatch(r'\d+[a-zA-Z]', token):
                result += self._tokenize_combined(token)
                last_token = token[-1]
                i += 1
            else:
                result.append(token)
                last_token = token if token not in ['<NUM>', '</NUM>'] else last_token
                i += 1
        return result

    def tokenize_latex(self, expression: str) -> List[str]:
        """Tách token cho biểu thức LaTeX."""
        expression = self.preprocess_latex(expression)
        tokens = [t for t in self._latex_re.findall(expression) if t.strip()]
        return self._tokenize_common(tokens, self._latex_re)

    def tokenize_vietnamese(self, text: str) -> List[str]:
        """Tách token cho văn bản tiếng Việt."""
        text = self.preprocess_text(text)
        tokens = [t for t in self._vietnamese_re.findall(text) if t.strip()]
        return self._tokenize_common(tokens, self._vietnamese_re)

    def tokenize(self, text: str, is_latex: bool = False) -> list[str]:
        """Tách text, chọn giữa LaTeX hoặc tiếng Việt."""
        if is_latex:
            return self.tokenize_latex(text)
        return self.tokenize_vietnamese(text)

    def build_vocab(self, data: str):
        counter = Counter()

        for item in data:
            # Tách equation (LaTeX)
            counter.update(self.tokenize(item['latex_equation'], is_latex=True))
            # Tách query (tiếng Việt)
            counter.update(self.tokenize(item['query']))
            # Tách solution steps (kết hợp LaTeX và tiếng Việt)
            for step in item['solution_steps']:
                parts = re.split(r'(\\[\(\[].*?\\[\)\]])', step)
                for part in parts:
                    if part.startswith('\\(') and part.endswith('\\)'):
                        latex_content = part[2:-2]
                        counter.update(self.tokenize(latex_content, is_latex=True))
                    else:
                        counter.update(self.tokenize(part))

            # Tách answer (tiếng Việt)
            counter.update(self.tokenize(item['answer']))

        # Khởi tạo vocab với special tokens
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        
        # Thêm token từ dataset
        for token, _ in counter.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.token_to_idx = self.vocab
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

        # Save vocab to vocab.txt
        # output_file = Path("src/mathsolver/checkpoints/vocab.txt")
        # self.save_vocab(output_file)

    def _encode_step(self, step: str) -> List[str]:
        """Mã hóa solution step."""
        tokens = ['<STEP>']
        parts = re.split(r'(\\[\(\[].*?\\[\)\]])', step)
        for part in parts:
            if part.startswith('\\(') and part.endswith('\\)'):
                tokens.append('\\(')
                tokens += self.tokenize(part[2:-2], is_latex=True)
                tokens.append('\\)')
            else:
                tokens += self.tokenize(part)
        tokens += ['</STEP>']
        return tokens

    def encode(self, latex_equation: str, query: str, solution_steps: list[str], answer: str) -> Tuple[List[int], List[int]]:
        # Input: <EQ>equation</EQ> <QUERY>query</QUERY>
        input_tokens = ['<sos>']
        input_tokens += ['<EQ>'] + self.tokenize(latex_equation, is_latex=True) + ['</EQ>']
        input_tokens += ['<QUERY>'] + self.tokenize(query) + ['</QUERY>']
        input_tokens += ['<eos>']

        # Output: <sos><STEP>step1</STEP> <STEP>step2</STEP> <ANSWER>answer</ANSWER><eos>
        output_tokens = ['<sos>']
        for step in solution_steps:
            output_tokens += self._encode_step(step)

        output_tokens += ['<ANSWER>'] + self.tokenize(answer) + ['</ANSWER>']
        output_tokens += ['<eos>']

        # Convert to IDs
        unk_id = self.token_to_idx['<unk>']
        input_ids = [self.token_to_idx.get(token, unk_id) for token in input_tokens]
        output_ids = [self.token_to_idx.get(token, unk_id) for token in output_tokens]

        return input_ids, output_ids

    def decode(self, token_ids: list[int]) -> str:
        """Giải mã token ID thành chuỗi, thêm dấu cách hợp lý."""
        tokens = [self.idx_to_token.get(idx, '<unk>') for idx in token_ids]
        if '<eos>' in tokens:
            tokens = tokens[:tokens.index('<eos>') + 1]

        result = []
        for i, token in enumerate(tokens):
            # Giữ nguyên các token đặc biệt
            if token in self.special_tokens:
                result.append(token)
            # Thêm khoảng trắng trước và sau toán tử hoặc dấu ngoặc
            elif token in ['+', '-', '*', '/', '=', '^', '(', ')', '\\cdot', '\\pm', '\\sqrt', '\\frac', '.', '{', '}']:
                result.append(f' {token} ')
            # Giữ nguyên lệnh LaTeX
            elif token.startswith('\\'):
                result.append(token)
            elif token in ['\\(', '\\)', '\\[', '\\]']:
                result.append(token)
            # Xử lý số và biến
            else:
                # Không thêm khoảng trắng nếu là số đi kèm biến (như 2x) hoặc trong <NUM>...</NUM>
                if i > 0 and tokens[i-1] in ['x', 'y', 'a', 'b', 'c'] and token.isdigit():
                    result.append(token)
                elif i > 0 and tokens[i-1] == '<NUM>':
                    result.append(token)
                else:
                    result.append(f' {token} ')

        # Chuẩn hóa khoảng trắng
        return re.sub(r'\s+', ' ', ''.join(result)).strip()
    
    def encode_for_test(self, latex_equation: str, query: str) -> list[int]:
        """Mã hóa input chỉ gồm equation + query (dùng khi test)."""
        input_tokens = ['<sos>']
        input_tokens += ['<EQ>'] + self.tokenize(latex_equation, is_latex=True) + ['</EQ>']
        input_tokens += ['<QUERY>'] + self.tokenize(query) + ['</QUERY>']
        input_tokens += ['<STEP>']

        unk_id = self.token_to_idx['<unk>']
        input_ids = [self.token_to_idx.get(token, unk_id) for token in input_tokens]

        return input_ids
    
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
    # Dataset mẫu
    dataset = [
       {
            "problem_type": "linear",
            "latex_equation": "-10x + -4 = -7x + -4",
            "query": "Tìm x sao cho -10x + -4 = -7x + -4",
            "solution_steps": [
                "Phương trình: \\(-10x + -4 = -7x + -4\\)",
                "Trừ hai vế cho \\(-7x\\): \\(-10x + -4 - -7x = -7x + -4 - -7x\\)",
                "Rút gọn: \\(-3x + -4 = -4\\)",
                "Trừ hai vế cho -4: \\(-3x + -4 - -4 = -4 - -4\\)",
                "Rút gọn: \\(-3x = 0\\)",
                "Chia hai vế cho -3: \\(x = \\frac{0}{-3}\\)",
                "Kết quả: \\(x = 0\\)"
            ],
            "answer": "x = 0"
        },
    ]

    json_folder = 'data/mathsolver'

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found in folder: {json_folder}")
    json_files = [os.path.join(json_folder, f) for f in json_files]

    all_samples = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_samples.extend(data)
    
    tokenizer = MathTokenizer()
    tokenizer.build_vocab(all_samples)
    print("Vocabulary size:", len(tokenizer.vocab))
    
    # In từ vựng để kiểm tra
    print("Vocabulary:", tokenizer.vocab)
    
    # Test mã hóa và giải mã
    for item in dataset:
        print(item['latex_equation'], item['query'], item['solution_steps'], item['answer'])
        input_ids, output_ids = tokenizer.encode(item['latex_equation'], item['query'], item['solution_steps'], item['answer'])
        print("\nInput IDs:", input_ids)
        print("Input decoded:", tokenizer.decode(input_ids))
        print("Output IDs:", output_ids)
        print("Output decoded:", tokenizer.decode(output_ids))