import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder  # `encoder` is an instance of EncoderLSTM
        self.decoder = decoder  # `decoder` is an instance of DecoderLSTM
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (src_len, batch_size, input_dim)
        # trg: (trg_len, batch_size, output_dim)
        
        trg_len, batch_size, output_dim = trg.shape
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        input = trg[0, :, :].unsqueeze(0)  # Initial input to the decoder (first timestep)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[t, :, :].unsqueeze(0) if teacher_force else output.unsqueeze(0)
        
        return outputs
