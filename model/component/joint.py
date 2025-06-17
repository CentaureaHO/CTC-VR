from typing import Optional

import torch
from torch import nn


class TransducerJoint(torch.nn.Module):

    def __init__(self,
                 vocab_size: int,
                 enc_output_size: int,
                 pred_output_size: int,
                 join_dim: int,
                 prejoin_linear: bool = True,
                 postjoin_linear: bool = False,
                 joint_mode: str = 'add',
                 activation: str = "tanh"):
        super().__init__()

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

        self.prejoin_linear = prejoin_linear
        self.postjoin_linear = postjoin_linear
        self.joint_mode = joint_mode

        if not self.prejoin_linear and not self.postjoin_linear:
            assert enc_output_size == pred_output_size == join_dim

        self.enc_ffn: Optional[nn.Linear] = None
        self.pred_ffn: Optional[nn.Linear] = None
        if self.prejoin_linear:
            self.enc_ffn = nn.Linear(enc_output_size, join_dim)
            self.pred_ffn = nn.Linear(pred_output_size, join_dim)

        self.post_ffn: Optional[nn.Linear] = None
        if self.postjoin_linear:
            self.post_ffn = nn.Linear(join_dim, join_dim)

        self.ffn_out = nn.Linear(join_dim, vocab_size)

    def forward(self,
                enc_out: torch.Tensor,
                pred_out: torch.Tensor,
                pre_project: bool = True) -> torch.Tensor:
        if (pre_project and self.prejoin_linear and self.enc_ffn is not None
                and self.pred_ffn is not None):
            enc_out = self.enc_ffn(enc_out)  # [B,T,E] -> [B,T,D]
            pred_out = self.pred_ffn(pred_out)  # [B,U,P] -> [B,U,D]

        if enc_out.ndim != 4:
            enc_out = enc_out.unsqueeze(2)  # [B,T,D] -> [B,T,1,D]
        if pred_out.ndim != 4:
            pred_out = pred_out.unsqueeze(1)  # [B,U,D] -> [B,1,U,D]

        out = enc_out + pred_out  # [B,T,U,D]

        if self.postjoin_linear and self.post_ffn is not None:
            out = self.post_ffn(out)

        out = self.activation(out)
        out = self.ffn_out(out)
        return out
