import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: (batch, seq_len, hidden_dim * 2)
        # hidden, cell: (num_layers * 2, batch, hidden_dim)
        return outputs, hidden, cell
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_dim) - trạng thái ẩn cuối cùng của decoder
        # encoder_outputs: (batch, seq_len, hidden_dim * 2)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, hidden_dim)
        energy = energy.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # (batch, 1, hidden_dim)
        attention = torch.bmm(v, energy).squeeze(1)  # (batch, seq_len)
        return F.softmax(attention, dim=1)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2 + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = Attention(hidden_dim)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch) - token trước đó
        # hidden, cell: (num_layers, batch, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim * 2)
        x = self.embedding(x).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden_dim * 2)
        
        lstm_input = torch.cat((x, context), dim=2)  # (batch, 1, hidden_dim + hidden_dim * 2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # (batch, output_dim)
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers)
        self.output_dim = output_dim

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch, seq_len, input_dim)
        # trg: (batch, trg_len)
        batch_size = src.size(0)
        trg_len = trg.size(1)
        outputs = torch.zeros(batch_size, trg_len, self.output_dim).to(src.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Combine forward and backward hidden states
        hidden = hidden.view(2, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -1, :, :], hidden[:, 0, :, :]), dim=2)
        hidden = hidden.sum(dim=0, keepdim=True)
        
        # Do the same for cell state
        cell = cell.view(2, 2, batch_size, -1)
        cell = torch.cat((cell[:, -1, :, :], cell[:, 0, :, :]), dim=2)
        cell = cell.sum(dim=0, keepdim=True)
        
        x = trg[:, 0]  # <sos> token

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            x = trg[:, t] if teacher_force else output.argmax(1)
        return outputs
    
# # Khởi tạo mô hình
# input_dim = 2  # (x, y) tọa độ
# output_dim = len(vocab)  # Kích thước vocabulary
# hidden_dim = 256
# model = Seq2Seq(input_dim, output_dim, hidden_dim)