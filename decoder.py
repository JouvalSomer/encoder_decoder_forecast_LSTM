import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=output_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        # x: (batch_size, 1, output_size)
        out, (hidden, cell) = self.lstm(x, (hidden, cell))  # out: (batch_size, 1, hidden_size)
        pred = self.fc(out.squeeze(1))                      # pred: (batch_size, output_size)
        return pred, hidden, cell