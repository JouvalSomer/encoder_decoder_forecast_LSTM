import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell