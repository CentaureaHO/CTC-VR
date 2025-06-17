import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
import sys
import os

from wenet.transformer.encoder import ConformerEncoder
from wenet.transducer.predictor import RNNPredictor
from model.component.joint import TransducerJoint
from model.component.transducer import add_blank, basic_greedy_search
from rnnt_common import Config
import torchaudio


class OnlineCTC(nn.Module):
    
    def __init__(self, vocab_size: int, encoder_output_size: int, 
                 dropout_rate: float = 0.0, blank_id: int = 0):
        super().__init__()
        self.ctc_lo = nn.Linear(encoder_output_size, vocab_size)
        self.dropout_rate = dropout_rate
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
    
    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ys_hat = self.ctc_lo(torch.nn.functional.dropout(hs_pad, p=self.dropout_rate))
        ys_hat = ys_hat.transpose(0, 1).log_softmax(2) # (T, B, V)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        ys_hat = ys_hat.transpose(0, 1) # (B, T, V)
        return loss, ys_hat
    
    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(self.ctc_lo(hs_pad), dim=2)
    
    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


class OnlineTransducer(nn.Module):
    
    def __init__(self, vocab_size: int, blank: int, encoder: nn.Module,
                 predictor: nn.Module, joint: nn.Module, ctc: Optional[nn.Module] = None,
                 ctc_weight: float = 0.3, ignore_id: int = -1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.blank = blank
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.transducer_weight = 1.0 - ctc_weight
        
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.ctc = ctc
    
    def forward(self, batch: dict, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        
        loss_rnnt = self._compute_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)
        loss = self.transducer_weight * loss_rnnt
        
        loss_ctc: Optional[torch.Tensor] = None
        if self.ctc_weight > 0.0 and self.ctc is not None:
            loss_ctc, _ = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
            loss = loss + self.ctc_weight * loss_ctc
        
        return {
            'loss': loss,
            'loss_ctc': loss_ctc,
            'loss_rnnt': loss_rnnt,
        }
    
    def streaming_forward(self, chunk_xs: torch.Tensor, chunk_lens: torch.Tensor,
                         offset: int, required_cache_size: int,
                         att_cache: torch.Tensor, cnn_cache: torch.Tensor,
                         predictor_states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        encoder_out, new_att_cache, new_cnn_cache = self.encoder.forward_chunk(
            chunk_xs, offset, required_cache_size, att_cache, cnn_cache
        )
        
        if predictor_states is None:
            predictor_states = self.predictor.init_state(
                batch_size=chunk_xs.size(0), 
                device=chunk_xs.device
            )
        
        return encoder_out, new_att_cache, new_cnn_cache, predictor_states
    
    def streaming_greedy_search(self, chunk_xs: torch.Tensor, chunk_lens: torch.Tensor,
                               offset: int, required_cache_size: int,
                               att_cache: torch.Tensor, cnn_cache: torch.Tensor,
                               predictor_states: Optional[List[torch.Tensor]] = None,
                               prev_token: int = None) -> Tuple[List[int], torch.Tensor, torch.Tensor, List[torch.Tensor], int]:
        encoder_out, new_att_cache, new_cnn_cache, new_predictor_states = self.streaming_forward(
            chunk_xs, chunk_lens, offset, required_cache_size, 
            att_cache, cnn_cache, predictor_states
        )
        
        if prev_token is None:
            prev_token = self.blank
        
        chunk_hyp = []
        current_states = new_predictor_states
        
        for t in range(encoder_out.size(1)):
            enc_out_t = encoder_out[:, t:t+1, :]  # [1, 1, D]
            
            pred_input = torch.tensor([[prev_token]], device=encoder_out.device)
            pred_out, current_states = self.predictor.forward_step(
                pred_input, 
                torch.zeros(1, 1, device=encoder_out.device),
                current_states
            )

            joint_out = self.joint(enc_out_t, pred_out)             # [1, 1, 1, V]
            joint_out = joint_out.squeeze(0).squeeze(0).squeeze(0)  # [V]

            pred_token = torch.argmax(joint_out).item()
            
            if pred_token != self.blank:
                chunk_hyp.append(pred_token)
                prev_token = pred_token
        
        return chunk_hyp, new_att_cache, new_cnn_cache, current_states, prev_token
    
    def _compute_rnnt_loss(self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor,
                          text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:

        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        
        predictor_out = self.predictor(ys_in_pad)
        
        joint_out = self.joint(encoder_out, predictor_out)
        
        rnnt_text = text.to(torch.int64)
        rnnt_text = torch.where(rnnt_text == self.ignore_id, 0, rnnt_text).to(torch.int32)
        rnnt_text_lengths = text_lengths.to(torch.int32)
        encoder_out_lens = encoder_out_lens.to(torch.int32)
        
        loss = torchaudio.functional.rnnt_loss(
            joint_out, rnnt_text, encoder_out_lens, rnnt_text_lengths,
            blank=self.blank, reduction="mean"
        )
        
        return loss


class OnlineRNNTModel(nn.Module):
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, 
                 vocab_size: int = 4336, blank_id: int = 0,
                 streaming: bool = True, static_chunk_size: int = 32,
                 use_dynamic_chunk: bool = True, ctc_weight: float = 0.3):
        super().__init__()
        
        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.streaming = streaming
        
        self.encoder = ConformerEncoder(
            input_size=input_dim,
            output_size=hidden_dim,
            attention_heads=4,
            linear_units=1024,
            num_blocks=12,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            input_layer="conv2d",
            pos_enc_layer_type="rel_pos",
            normalize_before=True,
            static_chunk_size=static_chunk_size if streaming else 0,
            use_dynamic_chunk=use_dynamic_chunk if streaming else False,
            use_dynamic_left_chunk=False,
            macaron_style=True,
            activation_type="swish",
            use_cnn_module=True,
            cnn_module_kernel=31,
            causal=True,
            cnn_module_norm="batch_norm"
        )
        
        self.predictor = RNNPredictor(
            voca_size=vocab_size,
            embed_size=hidden_dim,
            output_size=hidden_dim,
            embed_dropout=0.1,
            hidden_size=hidden_dim,
            num_layers=1,
            bias=True,
            rnn_type="lstm",
            dropout=0.1
        )
        
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
        
        self.ctc = OnlineCTC(
            vocab_size=vocab_size,
            encoder_output_size=hidden_dim,
            dropout_rate=0.1,
            blank_id=blank_id
        )
        
        self.transducer = OnlineTransducer(
            vocab_size=vocab_size,
            blank=blank_id,
            encoder=self.encoder,
            predictor=self.predictor,
            joint=self.joint,
            ctc=self.ctc,
            ctc_weight=ctc_weight,
            ignore_id=-1
        )

        self.reset_streaming_cache()
    
    def reset_streaming_cache(self):

        self.att_cache = torch.zeros((0, 0, 0, 0))
        self.cnn_cache = torch.zeros((0, 0, 0, 0))
        self.predictor_states = None
        self.prev_token = self.blank_id
        self.offset = 0
    
    def forward(self, audios, audio_lens, texts=None, text_lens=None):

        batch = {
            'feats': audios,
            'feats_lengths': audio_lens,
            'target': texts,
            'target_lengths': text_lens
        }
        
        current_device = audios.device
        
        if self.training and texts is not None:
            outputs = self.transducer(batch, current_device)
            loss = outputs['loss']
            loss_ctc = outputs.get('loss_ctc', None)
            loss_rnnt = outputs.get('loss_rnnt', None)
            return None, loss, {'loss_ctc': loss_ctc, 'loss_rnnt': loss_rnnt}
        else:
            if texts is not None:
                outputs = self.transducer(batch, current_device)
                loss = outputs['loss']
                loss_ctc = outputs.get('loss_ctc', None)
                loss_rnnt = outputs.get('loss_rnnt', None)
                return None, loss, {'loss_ctc': loss_ctc, 'loss_rnnt': loss_rnnt}
            else:
                if self.streaming:
                    return self.streaming_inference(audios, audio_lens)
                else:
                    encoder_out, encoder_mask = self.encoder(audios, audio_lens)
                    hyps = basic_greedy_search(self.transducer, encoder_out, 
                                             encoder_mask.squeeze(1).sum(1))
                    return hyps, None, None
    
    def streaming_inference(self, audios, audio_lens, 
                           chunk_size: Optional[int] = None) -> Tuple[List[List[int]], None, None]:
        if chunk_size is None:
            chunk_size = Config.static_chunk_size
        
        assert audios.size(0) == 1, "Streaming inference only supports batch size 1"
        
        self.reset_streaming_cache()
        
        device = audios.device
        self.att_cache = self.att_cache.to(device)
        self.cnn_cache = self.cnn_cache.to(device)
        
        audio_len = audio_lens[0].item()
        subsampling_rate = self.encoder.embed.subsampling_rate
        context = self.encoder.embed.right_context + 1
        stride = subsampling_rate * chunk_size
        chunk_window = (chunk_size - 1) * subsampling_rate + context
        
        full_hyp = []
        required_cache_size = chunk_size * Config.num_decoding_left_chunks
        
        for cur in range(0, audio_len - context + 1, stride):
            end = min(cur + chunk_window, audio_len)
            chunk_xs = audios[:, cur:end, :]
            chunk_lens = torch.tensor([end - cur], device=device)
            
            chunk_hyp, self.att_cache, self.cnn_cache, self.predictor_states, self.prev_token = \
                self.transducer.streaming_greedy_search(
                    chunk_xs, chunk_lens, self.offset, required_cache_size,
                    self.att_cache, self.cnn_cache, self.predictor_states, self.prev_token
                )
            
            full_hyp.extend(chunk_hyp)

            chunk_encoder_out_len = (end - cur - context) // subsampling_rate + 1
            self.offset += chunk_encoder_out_len
        
        return [full_hyp], None, None
    
    def process_chunk(self, chunk_audio: torch.Tensor) -> List[int]:
        chunk_lens = torch.tensor([chunk_audio.size(1)], device=chunk_audio.device)
        
        chunk_hyp, self.att_cache, self.cnn_cache, self.predictor_states, self.prev_token = \
            self.transducer.streaming_greedy_search(
                chunk_audio, chunk_lens, self.offset, 
                Config.static_chunk_size * Config.num_decoding_left_chunks,
                self.att_cache, self.cnn_cache, self.predictor_states, self.prev_token
            )
        
        subsampling_rate = self.encoder.embed.subsampling_rate
        context = self.encoder.embed.right_context + 1
        chunk_encoder_out_len = (chunk_audio.size(1) - context) // subsampling_rate + 1
        self.offset += chunk_encoder_out_len
        
        return chunk_hyp
    
    def ctc_greedy_search(self, audios, audio_lens):
        if self.streaming:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                audios, Config.static_chunk_size, Config.num_decoding_left_chunks
            )
        else:
            encoder_out, encoder_mask = self.encoder(audios, audio_lens)
        
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        
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
                if token != self.blank_id and token != prev_token:
                    hyp.append(token)
                prev_token = token
            hyps.append(hyp)
        
        return hyps
