import torch
import torch.nn as nn
import numpy as np

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder, target_len=10):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_len = target_len  # Set default target length
    
    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        if trg is not None:
            target_len = trg.size(1)
        else:
            target_len = self.target_len  # Use predefined target length during evaluation
        
        outputs = []
        
        # Encode the source sequence
        hidden, cell = self.encoder(src)
        
        # Initialize decoder input as zeros
        decoder_input = torch.zeros(batch_size, 1, self.decoder.output_size).to(src.device)
        
        for t in range(target_len):
            # Pass through the decoder
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(out.unsqueeze(1))  # Shape: (batch_size, 1, output_size)
            
            # Decide whether to use teacher forcing
            if trg is not None and np.random.rand() < teacher_forcing_ratio:
                # Use the actual next value as the next input
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                # Use the predicted value as the next input
                decoder_input = out.unsqueeze(1)
        
        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, target_len, output_size)
        return outputs
