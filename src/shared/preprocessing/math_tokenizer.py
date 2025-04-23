import re
from collections import Counter
from typing import List, Tuple, Callable, Optional
from torch_geometric.data import Data

class MathTokenizer:
    def __init__(self):
        # Token đặc biệt
        self.special_tokens = [
            '<pad>', '<sos>', '<eos>', '<unk>',
            '<NUM>', '</NUM>',
            '<EQ>', '</EQ>', '<QUERY>', '</QUERY>', '<STEP>', '</STEP>'
        ]
        self.operators = ['+', '-', '\\times', '\\cdot', '\\div', '=', '^', '\\pm', '<', '>', '\\leq', '\\geq']
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
            r'\\{|'                      # Mở nhóm LaTeX đặc biệt
            r'\\}|'                      # Đóng nhóm LaTeX đặc biệt
            r'\{|'                       # Mở nhóm
            r'\}|'                       # Đóng nhóm
            r'[0-9]|'                    # Từng chữ số
            r'\.|'                       # Dấu chấm riêng
            r'hoặc|và|là|'
            r'[a-zA-Z]+(?:\^\{.*?\})?|'  # Biến (như x, x^{2})
            r'[\+\-\*\/=^()]|'           # Toán tử
            r'\\[\+\-\*\/=^\\]|'         # Toán tử LaTeX (như \\pm)
            r'\S'                        # Ký tự khác
        )
        # Regex cho tiếng Việt: tách từ, ký tự đặc biệt, chữ số riêng lẻ
        self._vietnamese_re = re.compile(
            r'[^\s\d+\-*/=^()\\]+|'      # Từ tiếng V   iệt
            r'[\+\-\*\/=^()]|'           # Toán tử
            r'[0-9]|'                    # Từng chữ số
            r'\.|'                       # Dấu chấm riêng
            r'\S'                        # Ký tự khác
        )

    def preprocess_latex(self, expression: str) -> str:
        """Tiền xử lý biểu thức LaTeX để tách nhóm phức tạp."""
        # Thay thế các nhóm {content} để đảm bảo tách đúng
        expression = re.sub(r'\{([^\{\}]*?)\}', r' { \1 } ', expression)
        # Thêm khoảng trắng quanh toán tử và lệnh LaTeX
        expression = re.sub(r'(\\[a-zA-Z]+)', r' \1 ', expression)
        expression = re.sub(r'([\+\-\*\/=^()])', r' \1 ', expression)
        return re.sub(r'\s+', ' ', expression).strip()
    
    def preprocess_text(self, text: str) -> str:
        # Xử lý các dấu câu không nên dính với từ
        text = re.sub(r'([,.!?])', r' \1 ', text)
        text = re.sub(r':', ' : ', text)

        # Xử lý các trường hợp ký tự đặc biệt trong toán học như {, }, (, )
        text = re.sub(r'([{}()])', r' \1 ', text)  # Dấu ngoặc

        # Xử lý khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_latex(self, expression: str) -> list[str]:
        # Tiền xử lý trước khi tokenize
        expression = self.preprocess_latex(expression)
        tokens = [t for t in self._latex_re.findall(expression) if t.strip()]
        result = []
        last_token = None
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ['-', '+'] and i + 1 < len(tokens) and tokens[i + 1].isdigit():
                if last_token is None or last_token in ['(', '{'] + self.operators:
                    # Đây là số âm/dương
                    num_tokens = [token]
                    i += 1
                    while i < len(tokens) and (tokens[i].isdigit() or tokens[i] == '.'):
                        num_tokens.append(tokens[i])
                        i += 1
                    result += ['<NUM>'] + num_tokens + ['</NUM>']
                    last_token = '<NUM>'
                    continue  # tránh tăng i thêm nữa
                else:
                    # Đây là toán tử
                    result.append(token)
                    last_token = token
                    i += 1
            elif token.isdigit() or (token == '.' and i + 1 < len(tokens) and tokens[i + 1].isdigit()):
                num_tokens = [token]
                i += 1
                while i < len(tokens) and (tokens[i].isdigit() or tokens[i] == '.'):
                    num_tokens.append(tokens[i])
                    i += 1
                result += ['<NUM>'] + num_tokens + ['</NUM>']
                last_token = '<NUM>'
            elif re.fullmatch(r'\d+[a-zA-Z]', token):  # ví dụ 4x
                result += ['<NUM>', token[:-1], '</NUM>', token[-1]]
                last_token = token[-1]
                i += 1
            else:
                result.append(token)
                if token not in ['<NUM>', '</NUM>']:
                    last_token = token
                i += 1
        return result

    def tokenize_vietnamese(self, text: str) -> list[str]:
        text = text.lower()
        text = self.preprocess_text(text)
        return [t for t in self._vietnamese_re.findall(text) if t.strip()]

    def tokenize(self, text: str, is_latex: bool = False) -> list[str]:
        """Tách text, chọn giữa LaTeX hoặc tiếng Việt."""
        if is_latex:
            return self.tokenize_latex(text)
        return self.tokenize_vietnamese(text)
    
    def get_structure_features(self, tokens: List[str]) -> List[List[float]]:
        """
        Tạo đặc trưng cấu trúc cho các token.
        Trả về vector 7 chiều cho mỗi token:
        [is_number, is_variable, is_operator, is_bracket, is_latex_cmd, is_function, is_part_of_number]
        """
        features = []
        in_number = False
        for token in tokens:
            feat = [0.0] * 7
            
            if token in self.special_tokens:
                features.append(feat)
                if token == '<NUM>':
                    in_number = True
                elif token == '</NUM>':
                    in_number = False
                continue
            
            # Xác định đặc trưng
            # 1. is_number: token là chữ số hoặc dấu chấm
            if token in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']:
                feat[0] = 1.0
                if in_number:
                    feat[6] = 1.0  # Phần của số
                features.append(feat)
                continue
            
            # 2. is_variable: token là biến (x, y, f, hoặc biến có mũ như x^{2})
            if token in ['x', 'y', 'f', "f'", 'i'] or re.match(r'[a-zA-Z](?:\^\{.*?\})?', token):
                feat[1] = 1.0
                features.append(feat)
                continue
            
            # 3. is_operator: token là toán tử
            if token in self.operators:
                feat[2] = 1.0
                features.append(feat)
                continue

            if token in self.brackets:
                feat[3] = 1.0
                features.append(feat)
                continue
            
            # 5. is_latex_cmd: token là lệnh LaTeX
            if token in self.latex_commands:
                feat[4] = 1.0
                features.append(feat)
                continue

            # 6. is_function (trigonometric, log, etc.)
            if token in self.math_functions:
                feat[5] = 1.0
                features.append(feat)
                continue
            
            # Các token còn lại (tiếng Việt hoặc khác)
            features.append(feat)
        
        return features

    def build_vocab(self, dataset: list[dict]):
        counter = Counter()

        for item in dataset:
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

        # Khởi tạo vocab với special tokens
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        
        # Thêm token từ dataset
        for token, _ in counter.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.token_to_idx = self.vocab
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

    def encode(self, latex_equation: str, query: str, solution_steps: list[str],
               graph_fn: Optional[Callable[[str, List[str]], Data]] = None) -> Tuple[List[int], List[int], Optional[Data]]:
        # Input: <EQ>equation</EQ> <QUERY>query</QUERY>
        input_tokens = ['<sos>']
        input_tokens += ['<EQ>'] + self.tokenize(latex_equation, is_latex=True) + ['</EQ>']
        input_tokens += ['<QUERY>'] + self.tokenize(query) + ['</QUERY>']
        input_tokens += ['<eos>']

        # Output: <sos><STEP>step1</STEP> <STEP>step2</STEP><eos>
        output_tokens = ['<sos>']
        step_latex_expressions = []
        for step in solution_steps:
            output_tokens += ['<STEP>']
            parts = re.split(r'(\\[\(\[].*?\\[\)\]])', step)
            for part in parts:
                if part.startswith('\\(') and part.endswith('\\)'):
                    latex_content = part[2:-2]
                    output_tokens += self.tokenize(latex_content, is_latex=True)
                    step_latex_expressions.append(latex_content)
                else:
                    output_tokens += self.tokenize(part)
            output_tokens += ['</STEP>']
        output_tokens += ['<eos>']

        # Convert to IDs
        unk_id = self.token_to_idx['<unk>']
        input_ids = [self.token_to_idx.get(token, unk_id) for token in input_tokens]
        output_ids = [self.token_to_idx.get(token, unk_id) for token in output_tokens]

        graph_data = None
        if graph_fn:
            graph_data = graph_fn(step_latex_expressions)

        return input_ids, output_ids, graph_data

    def decode(self, token_ids: list[int]) -> str:
        """Giải mã token ID thành chuỗi, thêm dấu cách hợp lý."""
        tokens = [self.idx_to_token.get(idx, '<unk>') for idx in token_ids]
        if '<eos>' in tokens:
            tokens = tokens[:tokens.index('<eos>') + 1]

        result = []
        in_number = False
        for i, token in enumerate(tokens):
            if token == '<NUM>':
                in_number = True
                continue
            elif token == '</NUM>':
                in_number = False
                continue
            if token in self.special_tokens:
                result.append(f' {token} ')
            else:
                if token in ['+', '-', '*', '/', '=', '^', '(', ')', '\\cdot', '\\pm', '\\sqrt', '\\frac', '.', '{', '}']:
                    result.append(f' {token} ')
                elif token.startswith('\\'):
                    result.append(token)
                else:
                    # Xử lý số và biến
                    if in_number:
                        result.append(token)  # Không thêm khoảng trắng trong số
                    elif i > 0 and tokens[i-1] in ['x', 'y', 'a', 'b', 'c'] and token.isdigit():
                        result.append(token)  # ví dụ: 2x
                    else:
                        result.append(f' {token} ')

        # Loại bỏ dấu cách thừa
        decoded = ''.join(result).strip()
        decoded = re.sub(r'\s+', ' ', decoded)
        return decoded
    
    def encode_for_test(self, latex_equation: str, query: str) -> list[int]:
        """Mã hóa input chỉ gồm equation + query (dùng khi test)."""
        input_tokens = ['<sos>']
        input_tokens += ['<EQ>'] + self.tokenize(latex_equation, is_latex=True) + ['</EQ>']
        input_tokens += ['<QUERY>'] + self.tokenize(query) + ['</QUERY>']
        input_tokens += ['<STEP>']

        unk_id = self.token_to_idx['<unk>']
        input_ids = [self.token_to_idx.get(token, unk_id) for token in input_tokens]

        return input_ids

if __name__ == '__main__':
    # Dataset mẫu
    dataset = [
       {
            "latex_equation": "\\frac{2x + 31}{4} = 5",
            "query": "Tìm x",
            "solution_steps": [
                "Phương trình: \\(\\frac{2x + 31}{4} = 5\\)",
                "Nhân hai vế với 4: \\(2x + 31 = 20\\)",
                "Chuyển vế: \\(2x = -11\\)",
                "Chia hai vế cho 2: \\(x = -\\frac{11}{2}\\)"
            ]
        },
        {
            "latex_equation": "3(x - 2) = 9",
            "query": "Giải phương trình",
            "solution_steps": [
                "Phương trình: \\(3(x - 2) = 9\\)",
                "Mở ngoặc: \\(3x - 6 = 9\\)",
                "Chuyển vế: \\(3x = 15\\)",
                "Chia hai vế cho 3: \\(x = 5\\)"
            ]
        }
    ]
    
    tokenizer = MathTokenizer()
    tokenizer.build_vocab(dataset)
    
    # In từ vựng để kiểm tra
    print("Vocabulary:", tokenizer.vocab)
    
    # Test mã hóa và giải mã
    for item in dataset:
        input_ids, output_ids = tokenizer.encode(item['latex_equation'], item['query'], item['solution_steps'])
        print("\nInput IDs:", input_ids)
        print("Input decoded:", tokenizer.decode(input_ids))
        print("Output IDs:", output_ids)
        print("Output decoded:", tokenizer.decode(output_ids))

        test_ids = tokenizer.encode_for_test(item['latex_equation'], item['query'])
        print("Test Input IDs:", test_ids)
        print("Test Input decoded:", tokenizer.decode(test_ids))