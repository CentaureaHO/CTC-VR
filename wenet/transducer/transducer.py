from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch import nn

from wenet.transducer.predictor import PredictorBase
from wenet.transducer.search.greedy_search import basic_greedy_search
from wenet.transformer.ctc import CTC
from wenet.utils.common import IGNORE_ID, add_blank


class Transducer(nn.Module):
    """简化的Transducer模型，只支持RNNT+CTC混合训练"""

    def __init__(
        self,
        vocab_size: int,
        blank: int,
        encoder: nn.Module,
        predictor: PredictorBase,
        joint: nn.Module,
        ctc: Optional[CTC] = None,
        ctc_weight: float = 0.3,
        ignore_id: int = IGNORE_ID,
        transducer_weight: float = 0.7,
    ) -> None:
        super().__init__()
        
        assert ctc_weight + transducer_weight == 1.0
        
        self.vocab_size = vocab_size
        self.blank = blank
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.transducer_weight = transducer_weight
        
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.ctc = ctc

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """前向传播计算损失"""
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        
        # 编码器前向
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 计算RNNT损失
        loss_rnnt = self._compute_rnnt_loss(
            encoder_out,
            encoder_out_lens,
            text,
            text_lengths
        )

        loss = self.transducer_weight * loss_rnnt

        # 计算CTC损失
        loss_ctc: Optional[torch.Tensor] = None
        if self.ctc_weight != 0.0 and self.ctc is not None:
            loss_ctc, _ = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
            loss = loss + self.ctc_weight * loss_ctc.sum()

        return {
            'loss': loss,
            'loss_ctc': loss_ctc,
            'loss_rnnt': loss_rnnt,
        }

    def greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        n_steps: int = 64,
    ) -> List[List[int]]:
        """贪婪搜索解码"""
        assert speech.size(0) == 1
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        
        # 编码器前向
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum()
        
        # 贪婪搜索
        hyps = basic_greedy_search(
            self,
            encoder_out,
            encoder_out_lens,
            n_steps=n_steps
        )

        return hyps

    def _compute_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """计算RNNT损失"""
        # 在文本中添加blank符号
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        
        # 预测器前向
        predictor_out = self.predictor(ys_in_pad)
        
        # 联合网络前向
        joint_out = self.joint(encoder_out, predictor_out)
        
        # 准备RNNT损失计算的输入
        rnnt_text = text.to(torch.int64)
        rnnt_text = torch.where(rnnt_text == self.ignore_id, 0, rnnt_text).to(torch.int32)
        rnnt_text_lengths = text_lengths.to(torch.int32)
        encoder_out_lens = encoder_out_lens.to(torch.int32)
        
        # 计算RNNT损失
        loss = torchaudio.functional.rnnt_loss(
            joint_out,
            rnnt_text,
            encoder_out_lens,
            rnnt_text_lengths,
            blank=self.blank,
            reduction="mean"
        )
        
        return loss