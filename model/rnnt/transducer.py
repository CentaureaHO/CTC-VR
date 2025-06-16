from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rnnt.predictor import PredictorBase
from model.rnnt.search.greedy_search import basic_greedy_search


def add_blank(targets: torch.Tensor, blank: int, ignore_id: int) -> torch.Tensor:
    """
    在目标序列前添加blank
    """
    batch_size = targets.size(0)
    pad_targets = torch.ones(batch_size, 1, dtype=torch.long,
                           device=targets.device) * blank
    pad_targets = torch.cat([pad_targets, targets], dim=1)
    return pad_targets


class Transducer(nn.Module):
    """
    Transducer模型实现
    """
    def __init__(
        self,
        vocab_size: int,
        blank: int,
        encoder: nn.Module,
        predictor: PredictorBase,
        joint: nn.Module,
        ctc: Optional[nn.Module] = None,
        ignore_id: int = -1,
        transducer_weight: float = 1.0,
        ctc_weight: float = 0.0,
    ) -> None:
        assert transducer_weight + ctc_weight == 1.0
        super().__init__()

        self.blank = blank
        self.ignore_id = ignore_id
        self.vocab_size = vocab_size
        
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.ctc = ctc
        self.transducer_weight = transducer_weight
        self.ctc_weight = ctc_weight
        
        # 用于前缀束搜索
        self.bs = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        前向传播计算损失
        
        Args:
            speech: 语音特征 [B, T, D]
            speech_lengths: 特征长度 [B]
            text: 目标文本ID [B, U]
            text_lengths: 文本长度 [B]
            
        Returns:
            包含损失的字典
        """
        # 检查批大小一致
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)

        # 编码器前向传播
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 计算转录器损失
        loss_rnnt = self._compute_loss(encoder_out,
                                       encoder_out_lens,
                                       text,
                                       text_lengths)

        loss = self.transducer_weight * loss_rnnt

        # 可选的CTC损失
        loss_ctc = None
        if self.ctc_weight != 0.0 and self.ctc is not None:
            loss_ctc, _ = self.ctc(encoder_out, encoder_out_lens, text,
                                  text_lengths)
            loss = loss + self.ctc_weight * loss_ctc

        return {
            'loss': loss,
            'loss_rnnt': loss_rnnt,
            'loss_ctc': loss_ctc,
        }

    def _compute_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算转录器损失
        
        使用torchaudio的rnnt_loss实现
        """
        try:
            import torchaudio
        except ImportError:
            raise ImportError("torchaudio is required for computing RNNT loss")
            
        # 添加blank到目标序列前
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        
        # 预测器前向传播
        predictor_out = self.predictor(ys_in_pad)
        
        # 联合网络
        joint_out = self.joint(encoder_out, predictor_out)
        
        # 准备RNNT损失输入
        # 注意：某些损失实现要求pad是零
        # torch.int32 rnnt_loss required
        rnnt_text = text.to(torch.int64)
        rnnt_text = torch.where(rnnt_text == self.ignore_id, 0,
                               rnnt_text).to(torch.int32)
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

    def init_bs(self):
        """
        初始化用于束搜索的类
        """
        if self.bs is None:
            from model.rnnt.search.prefix_beam_search import PrefixBeamSearch
            self.bs = PrefixBeamSearch(self.encoder, self.predictor,
                                      self.joint, self.ctc, self.blank)

    def beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.3,
        transducer_weight: float = 0.7,
    ):
        """
        束搜索解码
        
        Args:
            speech: 语音特征 [B=1, T, D]
            speech_lengths: 特征长度 [B=1]
            decoding_chunk_size: 解码块大小
            beam_size: 束大小
            num_decoding_left_chunks: 左侧上下文块数量
            ctc_weight: CTC分数权重
            transducer_weight: 转录器分数权重
            
        Returns:
            最佳路径结果和分数
        """
        self.init_bs()
        beam, _ = self.bs.prefix_beam_search(
            speech,
            speech_lengths,
            decoding_chunk_size,
            beam_size,
            num_decoding_left_chunks,
            ctc_weight=ctc_weight,
            transducer_weight=transducer_weight,
        )
        return beam[0].hyp[1:], beam[0].score

    def greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        n_steps: int = 64,
    ) -> List[List[int]]:
        """
        贪婪搜索解码
        
        Args:
            speech: 语音特征 [B=1, T, D]
            speech_lengths: 特征长度 [B=1]
            decoding_chunk_size: 解码块大小
            num_decoding_left_chunks: 左侧上下文块数量
            n_steps: 每帧最大非空白标记数
            
        Returns:
            解码结果
        """
        assert speech.size(0) == 1
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        
        # 编码器前向传播
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum()
        
        # 执行贪婪搜索
        hyps = basic_greedy_search(self,
                                  encoder_out,
                                  encoder_out_lens,
                                  n_steps=n_steps)
        
        return hyps

    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        按块前向传播编码器（用于流式识别）
        """
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    def forward_predictor_step(
            self, xs: torch.Tensor, cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播预测器一步（用于流式识别）
        """
        assert len(cache) == 2
        # 虚拟padding
        padding = torch.zeros(1, 1, device=xs.device)
        return self.predictor.forward_step(xs, padding, cache)

    def forward_joint_step(self, enc_out: torch.Tensor,
                          pred_out: torch.Tensor) -> torch.Tensor:
        """
        前向传播联合网络一步（用于流式识别）
        """
        return self.joint(enc_out, pred_out)

    def forward_predictor_init_state(self, device=None) -> List[torch.Tensor]:
        """
        初始化预测器状态（用于流式识别）
        """
        if device is None:
            device = torch.device("cpu")
        return self.predictor.init_state(1, device=device)
