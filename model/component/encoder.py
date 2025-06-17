from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Conv2dSubsampling4(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_dim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(output_dim * ((input_dim - 1) // 2 - 1) // 2, output_dim),
            PositionalEncoding(output_dim, dropout_rate),
        )
        self.subsampling_rate = 4
        self.right_context = 0

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (B, 1, T, D)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out[0](x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.out[1](x), None
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class MultiHeadedAttention(nn.Module):

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float = 0.0):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(self, value: torch.Tensor, scores: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        attn = self.dropout(attn)
        x = torch.matmul(attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, pos_emb: torch.Tensor = None,
                cache: tuple = (torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0))) -> Tuple[torch.Tensor, tuple]:
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), (k, v)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, idim: int, hidden_units: int, dropout_rate: float, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class ConvolutionModule(nn.Module):

    def __init__(self, channels: int, kernel_size: int, activation: nn.Module = nn.SiLU(),
                 norm: str = "batch_norm", causal: bool = False, bias: bool = True):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1,
                                        padding=(kernel_size - 1) // 2, groups=channels, bias=bias)
        if norm == "batch_norm":
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor, mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)  # (batch, channels, time)

        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x = F.glu(x, dim=1)  # (batch, channels, time)

        x = self.depthwise_conv(x)
        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x)
        else:
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2), torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)


class ConformerEncoderLayer(nn.Module):

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module,
                 feed_forward_macaron: Optional[nn.Module] = None, conv_module: Optional[nn.Module] = None,
                 dropout_rate: float = 0.1, normalize_before: bool = True):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size)
        self.norm_mha = nn.LayerNorm(size)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size)
            self.norm_final = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor,
                mask_pad: torch.Tensor = torch.ones(
                    (0, 0, 0), dtype=torch.bool),
                att_cache: tuple = (torch.zeros((0, 0, 0, 0)),
                                    torch.zeros((0, 0, 0, 0))),
                cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0))) -> Tuple[torch.Tensor, torch.Tensor, tuple, torch.Tensor]:

        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * \
                self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    if max_len == 0:
        max_len = lengths.max().item()
    seq_range = torch.arange(
        0, max_len, dtype=torch.long, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand >= seq_length_expand


class ConformerEncoder(nn.Module):

    def __init__(self, input_size: int, output_size: int = 256, attention_heads: int = 4,
                 linear_units: int = 2048, num_blocks: int = 6, dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1, attention_dropout_rate: float = 0.0,
                 normalize_before: bool = True, positionwise_conv_kernel_size: int = 1,
                 macaron_style: bool = True, use_cnn_module: bool = True,
                 cnn_module_kernel: int = 15, causal: bool = False,
                 cnn_module_norm: str = "batch_norm", pos_enc_layer_type: str = "rel_pos",
                 static_chunk_size: int = 0, use_dynamic_chunk: bool = False):
        super().__init__()

        self._output_size = output_size
        self.embed = Conv2dSubsampling4(
            input_size, output_size, positional_dropout_rate)
        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk

        activation = nn.SiLU()
        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate),
                PositionwiseFeedForward(
                    output_size, linear_units, dropout_rate, activation),
                PositionwiseFeedForward(
                    output_size, linear_units, dropout_rate, activation) if macaron_style else None,
                ConvolutionModule(output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor,
                decoding_chunk_size: int = 0, num_decoding_left_chunks: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)

        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)

        for layer in self.encoders:
            xs, masks, _, _ = layer(xs, masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks
