from typing import Optional

import torch
from torch import nn
from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES


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

        self.activatoin = WENET_ACTIVATION_CLASSES[activation]()
        self.prejoin_linear = prejoin_linear
        self.postjoin_linear = postjoin_linear
        self.joint_mode = joint_mode

        if not self.prejoin_linear and not self.postjoin_linear:
            assert enc_output_size == pred_output_size == join_dim

        # 编码器和预测器的线性变换层
        self.enc_ffn: Optional[nn.Linear] = None
        self.pred_ffn: Optional[nn.Linear] = None
        if self.prejoin_linear:
            self.enc_ffn = nn.Linear(enc_output_size, join_dim)
            self.pred_ffn = nn.Linear(pred_output_size, join_dim)

        # 后处理线性层
        self.post_ffn: Optional[nn.Linear] = None
        if self.postjoin_linear:
            self.post_ffn = nn.Linear(join_dim, join_dim)

        # 输出层
        self.ffn_out = nn.Linear(join_dim, vocab_size)

    def forward(self,
                enc_out: torch.Tensor,
                pred_out: torch.Tensor,
                pre_project: bool = True) -> torch.Tensor:
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, T, P]
        Return:
            [B,T,U,V]
        """
        if (pre_project and self.prejoin_linear and self.enc_ffn is not None
                and self.pred_ffn is not None):
            enc_out = self.enc_ffn(enc_out)  # [B,T,E] -> [B,T,D]
            pred_out = self.pred_ffn(pred_out)
            
        if enc_out.ndim != 4:
            enc_out = enc_out.unsqueeze(2)  # [B,T,D] -> [B,T,1,D]
        if pred_out.ndim != 4:
            pred_out = pred_out.unsqueeze(1)  # [B,U,D] -> [B,1,U,D]

        # 联合操作（加法）
        out = enc_out + pred_out  # [B,T,U,V]

        if self.postjoin_linear and self.post_ffn is not None:
            out = self.post_ffn(out)

        # 激活函数和输出投影
        out = self.activatoin(out)
        out = self.ffn_out(out)
        return out