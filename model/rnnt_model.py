import torch
import torch.nn as nn
import torch.nn.functional as F
from model.component.transducer import Transducer
from model.component.predictor import RNNPredictor as Predictor
from model.component.joint import TransducerJoint
from wenet.transformer.encoder import ConformerEncoder
from typing import Tuple, Dict, Optional

class CTC(torch.nn.Module):
    """CTC module"""

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
        blank_id: int = 0,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
            blank_id: blank label.
        """
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id,
                                         reduction=reduction_type,
                                         zero_infinity=True)

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor,
                ys_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        ys_hat = ys_hat.transpose(0, 1)
        return loss, ys_hat

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

class TransducerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, blank_id, 
                 streaming=False, static_chunk_size=0, use_dynamic_chunk=False,
                 ctc_weight=0.3):
        super().__init__()
        
        # 编码器
        self.encoder = ConformerEncoder(
            input_size=input_dim,
            output_size=hidden_dim,
            attention_heads=4,
            linear_units=1024,
            num_blocks=12,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            positionwise_conv_kernel_size=1,
            macaron_style=True,
            use_cnn_module=True,
            cnn_module_kernel=31,
            causal=False,
            cnn_module_norm="layer_norm",
            pos_enc_layer_type="rel_pos",
            static_chunk_size=static_chunk_size,
            use_dynamic_chunk=use_dynamic_chunk
        )
        
        # 预测器
        self.predictor = Predictor(
            voca_size=vocab_size,
            embed_size=hidden_dim,
            output_size=hidden_dim,
            embed_dropout=0.1,
            hidden_size=hidden_dim,
            dropout=0.1,
            num_layers=1
        )

        # 联合网络
        self.joint = TransducerJoint(
            vocab_size=vocab_size,
            enc_output_size=hidden_dim,
            pred_output_size=hidden_dim,
            join_dim=hidden_dim,
            prejoin_linear=True,
            postjoin_linear=False,
            joint_mode='add',
            activation='tanh'
        )

        # CTC层
        self.ctc = CTC(
            odim=vocab_size,
            encoder_output_size=hidden_dim,
            dropout_rate=0.1,
            reduce=True,
            blank_id=blank_id
        )

        # Transducer模型
        self.transducer = Transducer(
            vocab_size=vocab_size,
            blank=blank_id,
            encoder=self.encoder,
            predictor=self.predictor,
            joint=self.joint,
            ctc=self.ctc,
            ctc_weight=ctc_weight,
            ignore_id=-1,
            transducer_weight=1.0-ctc_weight
        )
        
    def forward(self, audios, audio_lens, texts=None, text_lens=None):
        """前向传播"""
        batch = {
            'feats': audios,
            'feats_lengths': audio_lens,
            'target': texts,
            'target_lengths': text_lens
        }
        
        current_device = audios.device
        
        if self.training and texts is not None:
            # 训练模式
            outputs = self.transducer(batch, current_device)
            loss = outputs['loss']
            loss_ctc = outputs.get('loss_ctc', None)
            loss_rnnt = outputs.get('loss_rnnt', None)
            return None, loss, {'loss_ctc': loss_ctc, 'loss_rnnt': loss_rnnt}
        else:
            if texts is not None:
                # 验证模式，计算损失
                outputs = self.transducer(batch, current_device)
                loss = outputs['loss']
                loss_ctc = outputs.get('loss_ctc', None)
                loss_rnnt = outputs.get('loss_rnnt', None)
                return None, loss, {'loss_ctc': loss_ctc, 'loss_rnnt': loss_rnnt}
            else:
                # 推理模式，进行解码
                hyps = self.transducer.greedy_search(audios, audio_lens)
                scores = None 
                return hyps, scores, None

    def ctc_greedy_search(self, audios, audio_lens):
        """使用CTC进行贪婪搜索解码"""
        encoder_out, encoder_mask = self.encoder(audios, audio_lens)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        
        # CTC解码
        ctc_probs = self.ctc.log_softmax(encoder_out)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)
        topk_index = topk_index.squeeze(2)
        
        hyps = []
        for b in range(topk_index.size(0)):
            seq_len = encoder_out_lens[b].item()
            hyp = []
            prev_token = -1
            for t in range(seq_len):
                token = topk_index[b, t].item()
                if token != self.transducer.blank and token != prev_token:
                    hyp.append(token)
                prev_token = token
            hyps.append(hyp)
        
        return hyps