from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch import nn


def add_blank(text: torch.Tensor, blank: int, ignore_id: int) -> torch.Tensor:
    """在文本序列中添加blank符号"""
    batch_size, seq_len = text.size()
    new_seq_len = seq_len + 1

    # 创建新的序列，长度增加1
    ys_in = torch.zeros((batch_size, new_seq_len),
                        dtype=text.dtype, device=text.device)
    ys_in[:, 0] = blank  # 第一个位置是blank
    ys_in[:, 1:] = text  # 其余位置是原始文本

    return ys_in


def basic_greedy_search(
    model,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    n_steps: int = 64,
) -> List[List[int]]:
    batch_size = encoder_out.size(0)

    hyps = []
    for b in range(batch_size):
        hyp = []
        time_len = encoder_out_lens[b].item()
        enc_out_b = encoder_out[b:b+1, :time_len, :]  # [1, time_len, D]

        predictor_cache = model.predictor.init_state(1, device=encoder_out.device)
        prev_out_token = torch.tensor([[model.blank]], device=encoder_out.device, dtype=torch.long)

        for t in range(time_len):
            enc_out_t = enc_out_b[:, t:t+1, :]  # [1, 1, D] (当前时间步的编码器输出)

            for _ in range(n_steps):
                padding_for_step = torch.zeros_like(prev_out_token, device=prev_out_token.device)

                pred_out_u, new_predictor_cache = model.predictor.forward_step(
                    prev_out_token,
                    padding=padding_for_step,
                    cache=predictor_cache
                )  # pred_out_u: [1, 1, predictor_output_dim]

                joint_out = model.joint(enc_out_t, pred_out_u)  # [1, 1, 1, V]

                log_probs = joint_out.squeeze(0).squeeze(0).squeeze(0)  # [V]

                current_token_id = torch.argmax(log_probs).item()

                if current_token_id == model.blank:
                    break
                else:
                    hyp.append(current_token_id)
                    prev_out_token = torch.tensor([[current_token_id]], device=encoder_out.device, dtype=torch.long)
                    predictor_cache = new_predictor_cache

        hyps.append(hyp)

    return hyps


class Transducer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        blank: int,
        encoder: nn.Module,
        predictor: nn.Module,
        joint: nn.Module,
        ctc: Optional[nn.Module] = None,
        ctc_weight: float = 0.3,
        ignore_id: int = -1,
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
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)

        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        loss_rnnt = self._compute_rnnt_loss(
            encoder_out,
            encoder_out_lens,
            text,
            text_lengths
        )

        loss = self.transducer_weight * loss_rnnt

        loss_ctc: Optional[torch.Tensor] = None
        if self.ctc_weight != 0.0 and self.ctc is not None:
            loss_ctc, _ = self.ctc(
                encoder_out, encoder_out_lens, text, text_lengths)
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
        assert speech.size(0) == 1
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0

        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum()

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
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)

        predictor_out = self.predictor(ys_in_pad)

        joint_out = self.joint(encoder_out, predictor_out)

        rnnt_text = text.to(torch.int64)
        rnnt_text = torch.where(
            rnnt_text == self.ignore_id, 0, rnnt_text).to(torch.int32)
        rnnt_text_lengths = text_lengths.to(torch.int32)
        encoder_out_lens = encoder_out_lens.to(torch.int32)

        loss = torchaudio.functional.rnnt_loss(
            joint_out,
            rnnt_text,
            encoder_out_lens,
            rnnt_text_lengths,
            blank=self.blank,
            reduction="mean"
        )

        return loss
