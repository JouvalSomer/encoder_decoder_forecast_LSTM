import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers).to(self.device)
        
    def forward(self, src):
        # src: (src_len, batch_size, input_dim)
        outputs, (hidden, cell) = self.lstm(src)
        # outputs: (src_len, batch_size, hidden_dim)
        # hidden, cell: (num_layers, batch_size, hidden_dim)
        return hidden, cell
