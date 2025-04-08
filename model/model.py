import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from model.subsampling import Subsampling
from model.encoder import LSTMEncoder
import math

class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """

        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        else:  # for batched streaming decoding on GPU
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + \
                torch.arange(0, size).to(offset.device)  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb

class CTCModel(nn.Module):
    def __init__(self, in_dim, output_size,
                 vocab_size, blank_id
                ):
        super().__init__()
        self.subsampling = Subsampling(in_dim, output_size, subsampling_type=8)
        self.positional_encoding = PositionalEncoding(output_size, 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_size, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc_out = nn.Linear(output_size, vocab_size)

        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id,
                                         reduction="sum",
                                         zero_infinity=True)

    def forward(self, x, audio_lens, text, text_lens):
        x, encoder_out_lens = self.subsampling(x, audio_lens)
        x, pos_emb = self.positional_encoding(x, 0)
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
        x, pos_emb = self.positional_encoding(x, 0)
        x = self.encoder(x)
        predict = self.fc_out(x)
        predict = predict.log_softmax(2)

        return predict, encoder_out_lens

