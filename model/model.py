import torch
import torch.nn as nn
import torch.nn.functional as F
from model.subsampling import Subsampling
import math

class CTCModel(nn.Module):
    def __init__(self, in_dim, output_size,
                 vocab_size, blank_id
                ):
        super().__init__()
        self.subsampling = Subsampling(in_dim, output_size, subsampling_type=8)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_size, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc_out = nn.Linear(output_size, vocab_size)

        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id,
                                         reduction="sum",
                                         zero_infinity=True)

    def forward(self, x, audio_lens, text, text_lens):
        x, encoder_out_lens = self.subsampling(x, audio_lens)
        x = self.encoder(x)
        predict = self.fc_out(x)

        predict = predict.transpose(0, 1)
        predict = predict.log_softmax(2)
        loss = self.ctc_loss(predict, text, encoder_out_lens, text_lens)
        loss = loss / predict.size(1)
        predict = predict.transpose(0, 1)
        return predict, loss, encoder_out_lens
    
    def inference(self, x, audio_lens):
        x, encoder_out_lens = self.subsampling(x, audio_lens)
        x = self.encoder(x)
        predict = self.fc_out(x)
        predict = predict.log_softmax(2)

        return predict, encoder_out_lens

