# MathAI Project

This project is focused on developing a machine learning model for mathematical image recognition.

Link dataset: https://github.com/google-research/google-research/tree/master/mathwriting

Reference:

- https://actamachina.com/posts/handwritten-mathematical-expression-recognition
- https://github.com/ritheshkumar95/im2latex-tensorflow/tree/master

## Dataset

### Cấu trúc bộ dataset:

Bao gồm các thư mục train, test, valid, symbols, và synthetic. Mỗi thư mục chứa các file InkML đại diện cho nét viết tay (inks), với định dạng filename là [a-f0-9]{16}.inkml.

- **train**: Dữ liệu viết tay bởi con người, sao chép từ biểu thức LaTeX.
- **valid và test**: Dùng để kiểm tra hiệu suất mô hình.
- **symbols**: Các glyph đơn lẻ được trích xuất từ train.
- **synthetic**: Các nét viết được tạo tự động bằng cách ghép các glyph từ symbols dựa trên bounding box từ LaTeX.

### File InkML:

Lưu thông tin nét viết (tọa độ không gian, thời gian tương đối) và metadata như label (biểu thức LaTeX gốc), normalizedLabel (biểu thức chuẩn hóa), splitTagOriginal, sampleId, inkCreationMethod, và labelCreationMethod.

### File phụ trợ:

- **synthetic-bboxes.jsonl**: Chứa thông tin bounding box từ LaTeX để tạo dữ liệu synthetic.
- **symbols.jsonl**: Tham chiếu đến các glyph đơn lẻ từ train để tạo symbols.

### Đặc điểm:

- Tọa độ không được chuẩn hóa (phụ thuộc thiết bị).
- Không có thông tin phân đoạn hay dữ liệu nhân khẩu học.
- normalizedLabel được khuyến nghị làm ground truth cho việc huấn luyện.

## Install

1. Tạo và kích hoạt môi trường ảo:

   ```bash
   python -m venv math_env
   source math_env/bin/activate  # Trên Windows: math_env\Scripts\activate
   ```

2. Cài đặt các thư viện:
   ```bash
   pip install -r requirements.txt
   ```
