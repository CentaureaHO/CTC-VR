import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from model.subsampling import Subsampling
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

class RelPositionalEncoding(torch.nn.Module):
    def __init__(self,
                d_model: int,
                dropout_rate: float,
                max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.relative_pe = torch.zeros(2*self.max_len-1, self.d_model)
        rel_position = torch.arange(-self.max_len+1, self.max_len, dtype=torch.float32).unsqueeze(1)
        self.relative_pe[:, 0::2] = torch.sin(rel_position * div_term)
        self.relative_pe[:, 1::2] = torch.cos(rel_position * div_term)
        
        self.register_buffer("pe_buffer", self.pe.unsqueeze(0))
        self.register_buffer("rel_pe_buffer", self.relative_pe)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)

        if isinstance(offset, int):
            pos_emb = self.pe_buffer[:, offset:offset+seq_len]
        else:
            pos_emb = self.pe_buffer[:, :seq_len]
        
        center = self.max_len - 1
        start = center - seq_len + 1
        end = center + seq_len
        rel_pos_emb = self.rel_pe_buffer[start:end]
        
        x = x * self.xscale + pos_emb
        
        return self.dropout(x), rel_pos_emb

class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31, expansion_factor=2, dropout=0.1):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, inner_dim, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [B, T, C]
        residual = x
        
        x = self.layer_norm(x)
        
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, C]
        
        return x + residual

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ff_expansion=4, conv_expansion=2, 
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()

        self.ff1_norm = nn.LayerNorm(dim)
        self.ff1_linear1 = nn.Linear(dim, dim * ff_expansion)
        self.ff1_activation = nn.SiLU()
        self.ff1_dropout1 = nn.Dropout(dropout)
        self.ff1_linear2 = nn.Linear(dim * ff_expansion, dim)
        self.ff1_dropout2 = nn.Dropout(dropout)
        
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        
        self.conv = ConformerConvModule(dim, conv_kernel_size, conv_expansion, dropout)

        self.ff2_norm = nn.LayerNorm(dim)
        self.ff2_linear1 = nn.Linear(dim, dim * ff_expansion)
        self.ff2_activation = nn.SiLU()
        self.ff2_dropout1 = nn.Dropout(dropout)
        self.ff2_linear2 = nn.Linear(dim * ff_expansion, dim)
        self.ff2_dropout2 = nn.Dropout(dropout)
        
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # FFN 1
        ff1_out = self.ff1_norm(x)
        ff1_out = self.ff1_linear1(ff1_out)
        ff1_out = self.ff1_activation(ff1_out)
        ff1_out = self.ff1_dropout1(ff1_out)
        ff1_out = self.ff1_linear2(ff1_out)
        ff1_out = self.ff1_dropout2(ff1_out)
        x = x + 0.5 * ff1_out
        
        # Self Attention
        residual = x
        x_norm = self.attn_norm(x)
        x_t = x_norm.transpose(0, 1)                # [T, B, C]
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x = residual + attn_out.transpose(0, 1)     # [B, T, C]
        
        # Conv Module
        x = x + self.conv(x)
        
        # FFN 2
        ff2_out = self.ff2_norm(x)
        ff2_out = self.ff2_linear1(ff2_out)
        ff2_out = self.ff2_activation(ff2_out)
        ff2_out = self.ff2_dropout1(ff2_out)
        ff2_out = self.ff2_linear2(ff2_out)
        ff2_out = self.ff2_dropout2(ff2_out)
        x = x + 0.5 * ff2_out
        
        return self.final_norm(x)

class ConformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(
                dim=d_model, 
                num_heads=num_heads, 
                ff_expansion=d_ff // d_model, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class CTCModel(nn.Module):
    def __init__(self, in_dim, output_size,
                 vocab_size, blank_id
                ):
        super().__init__()
        self.subsampling = Subsampling(in_dim, output_size, subsampling_type=8)
        
        self.positional_encoding = RelPositionalEncoding(output_size, 0.1)
        
        self.encoder = ConformerEncoder(
            d_model=output_size,
            num_heads=8,
            num_layers=3,
            d_ff=output_size * 4,
            dropout=0.1
        )

        self.fc_out = nn.Linear(output_size, vocab_size)

        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id,
                                         reduction="sum",
                                         zero_infinity=True)

    def forward(self, x, audio_lens, text, text_lens):
        x, encoder_out_lens = self.subsampling(x, audio_lens)
        x, _ = self.positional_encoding(x, 0)
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
        x, _ = self.positional_encoding(x, 0)
        x = self.encoder(x)
        predict = self.fc_out(x)
        predict = predict.log_softmax(2)

        return predict, encoder_out_lens
