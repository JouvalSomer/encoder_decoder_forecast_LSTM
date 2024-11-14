import torch
import torch.nn as nn

class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input, hidden, cell):
        # input: (1, batch_size, output_dim)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # output: (1, batch_size, hidden_dim)
        prediction = self.fc_out(output.squeeze(0))  # prediction: (batch_size, output_dim)
        
        return prediction, hidden, cell
