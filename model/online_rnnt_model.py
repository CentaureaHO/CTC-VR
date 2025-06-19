import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
import sys
import os

from wenet.transformer.encoder import ConformerEncoder
from wenet.transducer.predictor import RNNPredictor
from model.component.joint import TransducerJoint
from model.component.transducer import add_blank, basic_greedy_search
import torchaudio


class OnlineCTC(nn.Module):

    def __init__(self, vocab_size: int, encoder_output_size: int,
                 dropout_rate: float = 0.0, blank_id: int = 0):
        super().__init__()
        self.ctc_lo = nn.Linear(encoder_output_size, vocab_size)
        self.dropout_rate = dropout_rate
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id, reduction="mean", zero_infinity=True)

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ys_hat = self.ctc_lo(torch.nn.functional.dropout(
            hs_pad, p=self.dropout_rate))
        ys_hat = ys_hat.transpose(0, 1).log_softmax(2)  # (T, B, V)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        ys_hat = ys_hat.transpose(0, 1)  # (B, T, V)
        return loss, ys_hat

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


class BeamHypothesis:
    def __init__(self, tokens: List[int], log_prob: float, predictor_states: Optional[List[torch.Tensor]] = None):
        self.tokens = tokens
        self.log_prob = log_prob
        self.predictor_states = predictor_states

    def __lt__(self, other):
        return self.log_prob < other.log_prob

    def copy(self):
        return BeamHypothesis(
            tokens=self.tokens.copy(),
            log_prob=self.log_prob,
            predictor_states=self.predictor_states
        )


class OnlineRNNTModel(nn.Module):

    def __init__(self,
                 input_dim: int = 80,
                 hidden_dim: int = 256,
                 vocab_size: int = 4336,
                 blank_id: int = 0,
                 streaming: bool = True,
                 static_chunk_size: int = 32,
                 use_dynamic_chunk: bool = True,
                 ctc_weight: float = 0.3,
                 predictor_layers: int = 1,
                 predictor_dropout: float = 0.1,
                 ctc_dropout_rate: float = 0.1,
                 rnnt_loss_clamp: float = -1.0,
                 ignore_id: int = -1
                 ):
        super().__init__()

        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.streaming = streaming
        self.ctc_weight = ctc_weight
        self.ignore_id = ignore_id
        self.rnnt_loss_clamp = rnnt_loss_clamp

        self.encoder_output_size = hidden_dim
        self.encoder = ConformerEncoder(
            input_size=input_dim,
            output_size=self.encoder_output_size,
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
            embed_dropout=predictor_dropout,
            hidden_size=hidden_dim,
            num_layers=predictor_layers,
            bias=True,
            rnn_type="lstm",
            dropout=predictor_dropout
        )

        self.joint = TransducerJoint(
            vocab_size=vocab_size,
            enc_output_size=self.encoder_output_size,
            pred_output_size=hidden_dim,
            join_dim=hidden_dim
        )

        if self.ctc_weight > 0.0:
            self.ctc_head = OnlineCTC(
                vocab_size=self.vocab_size,
                encoder_output_size=self.encoder_output_size,
                dropout_rate=ctc_dropout_rate,
                blank_id=self.blank_id
            )
        else:
            self.ctc_head = None

        self.streaming_att_cache: Optional[torch.Tensor] = None
        self.streaming_cnn_cache: Optional[torch.Tensor] = None
        self.streaming_predictor_states: Optional[List[torch.Tensor]] = None
        self.streaming_last_emitted_token: int = self.blank_id

        self.streaming_beam_hypotheses: Optional[List[BeamHypothesis]] = None

    def reset_streaming_cache(self, device: Optional[torch.device] = None):
        if hasattr(self.encoder, 'reset_streaming_state'):
            self.encoder.reset_streaming_state()
        elif hasattr(self.encoder, 'reset_streaming_cache'):
            self.encoder.reset_streaming_cache()

        current_device = device if device is not None else next(
            self.parameters()).device
        self.streaming_att_cache = torch.zeros(
            (0, 0, 0, 0), device=current_device)
        self.streaming_cnn_cache = torch.zeros(
            (0, 0, 0, 0), device=current_device)

        self.streaming_predictor_states = None

        self.streaming_last_emitted_token = self.blank_id

        self.streaming_beam_hypotheses = None

        self._global_encoder_offset = 0

    def _decode_chunk_streaming_logic(self,
                                      chunk_xs: torch.Tensor,
                                      offset: int,
                                      required_cache_size: int,
                                      att_cache_in: torch.Tensor,
                                      cnn_cache_in: torch.Tensor,
                                      predictor_states_in: Optional[List[torch.Tensor]],
                                      prev_token_in: int,
                                      n_steps: int = 10) -> Tuple[List[int], torch.Tensor, torch.Tensor, List[torch.Tensor], int]:
        encoder_out, att_cache_out, cnn_cache_out = self.encoder.forward_chunk(
            xs=chunk_xs,
            offset=offset,
            required_cache_size=required_cache_size,
            att_cache=att_cache_in,
            cnn_cache=cnn_cache_in
        )  # encoder_out: [1, chunk_T_encoder, D_encoder]

        current_predictor_states = predictor_states_in
        if current_predictor_states is None:
            current_predictor_states = self.predictor.init_state(
                batch_size=chunk_xs.size(0), device=chunk_xs.device
            )

        current_pred_input_token_val = prev_token_in

        chunk_hyp_list = []

        for t in range(encoder_out.size(1)):
            enc_out_t = encoder_out[:, t:t+1, :]  # [1, 1, D_encoder]

            for _ in range(n_steps):
                pred_input_tensor = torch.tensor(
                    [[current_pred_input_token_val]], device=encoder_out.device, dtype=torch.long)
                padding_for_step = torch.zeros_like(
                    pred_input_tensor, device=pred_input_tensor.device)

                # pred_out_u: [1, 1, D_predictor_out]
                pred_out_u, next_predictor_states_after_step = self.predictor.forward_step(
                    input=pred_input_tensor,
                    padding=padding_for_step,
                    cache=current_predictor_states
                )

                joint_out = self.joint(enc_out_t, pred_out_u)  # [1, 1, 1, V]
                log_probs = joint_out.squeeze(0).squeeze(0).squeeze(0)  # [V]

                emitted_token_id = torch.argmax(log_probs).item()

                if emitted_token_id == self.blank_id:
                    # Blank emitted for this t && move to next t.
                    break
                else:
                    chunk_hyp_list.append(emitted_token_id)
                    current_pred_input_token_val = emitted_token_id
                    current_predictor_states = next_predictor_states_after_step

        return chunk_hyp_list, att_cache_out, cnn_cache_out, current_predictor_states, current_pred_input_token_val

    def forward(self, audios, audio_lens, texts=None, text_lens=None):
        current_device = next(self.parameters()).device

        if not self.streaming or texts is not None:
            audios, audio_lens = audios.to(
                current_device), audio_lens.to(current_device)
            if texts is not None:
                texts, text_lens = texts.to(
                    current_device), text_lens.to(current_device)

            encoder_out, encoder_mask = self.encoder(audios, audio_lens)
            encoder_out_lens = encoder_mask.squeeze(1).sum(1)

            loss_dict = {}
            total_loss = torch.tensor(0.0, device=current_device)

            if texts is not None:
                ys_in_pad = add_blank(texts, self.blank_id, self.ignore_id)
                predictor_out = self.predictor(ys_in_pad)
                joint_out = self.joint(encoder_out, predictor_out)

                rnnt_text_targets = texts.to(torch.int32)

                loss_rnnt = torchaudio.functional.rnnt_loss(
                    logits=joint_out,
                    targets=rnnt_text_targets,
                    logit_lengths=encoder_out_lens.to(torch.int32),
                    target_lengths=text_lens.to(torch.int32),
                    blank=self.blank_id,
                    reduction="mean",
                    clamp=self.rnnt_loss_clamp
                )
                loss_dict['loss_rnnt'] = loss_rnnt.item()
                total_loss = (1.0 - self.ctc_weight) * loss_rnnt

            if self.ctc_weight > 0.0 and self.ctc_head is not None and texts is not None:
                loss_ctc, _ = self.ctc_head(
                    encoder_out, encoder_out_lens, texts, text_lens)
                loss_dict['loss_ctc'] = loss_ctc.item()
                total_loss = total_loss + self.ctc_weight * loss_ctc

            if texts is not None:
                return joint_out, total_loss, loss_dict
            else:
                hyps = basic_greedy_search(self, encoder_out, encoder_out_lens)
                return hyps, None, None

        else:
            return self.streaming_inference(audios.to(current_device), audio_lens.to(current_device))

    def streaming_inference(self, audios: torch.Tensor, audio_lens: torch.Tensor,
                            chunk_size_ms: Optional[int] = None) -> Tuple[List[List[int]], None, None]:
        assert self.streaming, "Model is not in streaming mode for streaming_inference."
        assert audios.size(
            0) == 1, "Streaming inference currently supports batch size 1 only."

        current_device = audios.device
        self.reset_streaming_cache(device=current_device)

        subsampling_rate = self.encoder.embed.subsampling_rate if hasattr(
            self.encoder.embed, 'subsampling_rate') else 4

        encoder_static_chunk_frames = self.encoder.static_chunk_size if hasattr(
            self.encoder, 'static_chunk_size') and self.encoder.static_chunk_size > 0 else 16

        input_audio_chunk_frames = encoder_static_chunk_frames * subsampling_rate

        if chunk_size_ms is not None:
            input_audio_chunk_frames = int(chunk_size_ms / 10)

        num_input_audio_frames = audio_lens.item()
        full_hyp = []

        current_input_offset_frames = 0

        min_input_audio_frames_for_conv = max(16, subsampling_rate * 4)

        if input_audio_chunk_frames < min_input_audio_frames_for_conv:
            if num_input_audio_frames >= min_input_audio_frames_for_conv:
                input_audio_chunk_frames = min_input_audio_frames_for_conv
            else:
                if num_input_audio_frames < 7:
                    print(
                        f"Error: Input audio too short ({num_input_audio_frames} frames) for conv layers. Skipping.")
                    return [[] for _ in range(audios.size(0))], None, None
                input_audio_chunk_frames = num_input_audio_frames

        while current_input_offset_frames < num_input_audio_frames:
            chunk_end_input_frames = min(
                current_input_offset_frames + input_audio_chunk_frames, num_input_audio_frames)
            current_chunk_audio = audios[:,
                                         current_input_offset_frames:chunk_end_input_frames, :]

            if current_chunk_audio.size(1) == 0:
                break

            if current_chunk_audio.size(1) < 7:
                current_input_offset_frames = chunk_end_input_frames
                continue

            encoder_output_offset = current_input_offset_frames // subsampling_rate
            required_enc_cache_size = encoder_output_offset

            chunk_hyp_list, \
                self.streaming_att_cache, self.streaming_cnn_cache, \
                self.streaming_predictor_states, self.streaming_last_emitted_token = self._decode_chunk_streaming_logic(
                    chunk_xs=current_chunk_audio,
                    offset=encoder_output_offset,
                    required_cache_size=required_enc_cache_size,
                    att_cache_in=self.streaming_att_cache,
                    cnn_cache_in=self.streaming_cnn_cache,
                    predictor_states_in=self.streaming_predictor_states,
                    prev_token_in=self.streaming_last_emitted_token
                )

            if chunk_hyp_list:
                full_hyp.extend(chunk_hyp_list)

            current_input_offset_frames = chunk_end_input_frames

        return [full_hyp], None, None

    def process_single_chunk(self, chunk_audio: torch.Tensor, chunk_len: torch.Tensor) -> Tuple[List[int], None, None]:
        assert self.streaming, "Model is not in streaming mode for process_single_chunk."
        assert chunk_audio.size(
            0) == 1, "Single chunk processing currently supports batch size 1 only."

        current_device = chunk_audio.device

        if self.streaming_att_cache is None or self.streaming_cnn_cache is None:
            self.reset_streaming_cache(device=current_device)

        if chunk_audio.size(1) < 7:
            print(
                f"Warning: Chunk too small ({chunk_audio.size(1)} frames), skipping")
            return [], None, None

        subsampling_rate = self.encoder.embed.subsampling_rate if hasattr(
            self.encoder.embed, 'subsampling_rate') else 4

        if hasattr(self, '_global_encoder_offset'):
            encoder_output_offset = self._global_encoder_offset
        else:
            self._global_encoder_offset = 0
            encoder_output_offset = 0

        required_enc_cache_size = encoder_output_offset

        chunk_hyp_list, \
            self.streaming_att_cache, self.streaming_cnn_cache, \
            self.streaming_predictor_states, self.streaming_last_emitted_token = self._decode_chunk_streaming_logic(
                chunk_xs=chunk_audio,
                offset=encoder_output_offset,
                required_cache_size=required_enc_cache_size,
                att_cache_in=self.streaming_att_cache,
                cnn_cache_in=self.streaming_cnn_cache,
                predictor_states_in=self.streaming_predictor_states,
                prev_token_in=self.streaming_last_emitted_token
            )

        estimated_encoder_frames = chunk_audio.size(1) // subsampling_rate
        self._global_encoder_offset += estimated_encoder_frames

        return chunk_hyp_list, None, None

    def _decode_chunk_beam_search(self,
                                  chunk_xs: torch.Tensor,
                                  offset: int,
                                  required_cache_size: int,
                                  att_cache_in: torch.Tensor,
                                  cnn_cache_in: torch.Tensor,
                                  beam_hypotheses_in: Optional[List[BeamHypothesis]],
                                  beam_size: int = 4,
                                  n_steps: int = 10) -> Tuple[List[BeamHypothesis], torch.Tensor, torch.Tensor]:

        encoder_out, att_cache_out, cnn_cache_out = self.encoder.forward_chunk(
            xs=chunk_xs,
            offset=offset,
            required_cache_size=required_cache_size,
            att_cache=att_cache_in,
            cnn_cache=cnn_cache_in
        )

        if beam_hypotheses_in is None:
            initial_states = self.predictor.init_state(
                batch_size=chunk_xs.size(0), device=chunk_xs.device
            )
            beam_hypotheses_in = [BeamHypothesis(
                tokens=[],
                log_prob=0.0,
                predictor_states=self._deep_copy_states(initial_states)
            )]

        current_beam = beam_hypotheses_in

        for t in range(encoder_out.size(1)):
            enc_out_t = encoder_out[:, t:t+1, :]  # [1, 1, D_encoder]

            all_candidates = []

            for hyp_idx, hyp in enumerate(current_beam):
                temp_tokens = hyp.tokens.copy()
                temp_log_prob = hyp.log_prob
                temp_pred_states = self._deep_copy_states(hyp.predictor_states)

                last_emitted_token = temp_tokens[-1] if temp_tokens else self.blank_id

                step_candidates = []

                for step in range(n_steps):
                    pred_input_tensor = torch.tensor(
                        [[last_emitted_token]], device=encoder_out.device, dtype=torch.long)
                    padding_for_step = torch.zeros_like(pred_input_tensor)

                    pred_out_u, next_pred_states = self.predictor.forward_step(
                        input=pred_input_tensor,
                        padding=padding_for_step,
                        cache=temp_pred_states
                    )

                    joint_out = self.joint(
                        enc_out_t, pred_out_u)  # [1, 1, 1, V]
                    log_probs = torch.log_softmax(
                        joint_out.squeeze(), dim=-1)  # [V]

                    blank_log_prob = log_probs[self.blank_id].item()

                    blank_candidate = BeamHypothesis(
                        tokens=temp_tokens.copy(),
                        log_prob=temp_log_prob + blank_log_prob,
                        predictor_states=self._deep_copy_states(
                            temp_pred_states)
                    )
                    step_candidates.append(blank_candidate)

                    non_blank_mask = torch.ones_like(
                        log_probs, dtype=torch.bool)
                    non_blank_mask[self.blank_id] = False
                    non_blank_log_probs = log_probs[non_blank_mask]
                    non_blank_indices = torch.arange(log_probs.size(
                        0), device=log_probs.device)[non_blank_mask]

                    if non_blank_log_probs.numel() > 0:
                        k = min(beam_size, non_blank_log_probs.size(0))
                        topk_log_probs, topk_idx = torch.topk(
                            non_blank_log_probs, k)
                        topk_tokens = non_blank_indices[topk_idx]

                        for i in range(k):
                            token_id = topk_tokens[i].item()
                            token_log_prob = topk_log_probs[i].item()

                            new_tokens = temp_tokens + [token_id]
                            new_log_prob = temp_log_prob + token_log_prob
                            new_candidate = BeamHypothesis(
                                tokens=new_tokens,
                                log_prob=new_log_prob,
                                predictor_states=self._deep_copy_states(
                                    next_pred_states)
                            )
                            step_candidates.append(new_candidate)

                    if blank_log_prob >= log_probs.max().item() - 1e-6:
                        break

                    if non_blank_log_probs.numel() > 0:
                        best_non_blank_idx = torch.argmax(non_blank_log_probs)
                        best_token = non_blank_indices[best_non_blank_idx].item(
                        )
                        best_log_prob = non_blank_log_probs[best_non_blank_idx].item(
                        )

                        temp_tokens.append(best_token)
                        temp_log_prob += best_log_prob
                        temp_pred_states = next_pred_states
                        last_emitted_token = best_token
                    else:
                        break

                all_candidates.extend(step_candidates)

            if all_candidates:
                all_candidates.sort(key=lambda x: x.log_prob, reverse=True)

                unique_candidates = []
                seen_tokens = set()
                for cand in all_candidates:
                    tokens_tuple = tuple(cand.tokens)
                    if tokens_tuple not in seen_tokens:
                        unique_candidates.append(cand)
                        seen_tokens.add(tokens_tuple)
                        if len(unique_candidates) >= beam_size:
                            break

                current_beam = unique_candidates[:beam_size]
            else:
                pass

        return current_beam, att_cache_out, cnn_cache_out

    def _deep_copy_states(self, states):
        if states is None:
            return None
        if isinstance(states, (list, tuple)):
            return [s.clone() if torch.is_tensor(s) else s for s in states]
        elif torch.is_tensor(states):
            return states.clone()
        else:
            return states

    def streaming_beam_search(self, audios: torch.Tensor, audio_lens: torch.Tensor,
                              beam_size: int = 4, chunk_size_ms: Optional[int] = None) -> Tuple[List[List[int]], None, None]:
        assert self.streaming, "Model is not in streaming mode for streaming_beam_search."
        assert audios.size(
            0) == 1, "Streaming beam search currently supports batch size 1 only."

        current_device = audios.device
        self.reset_streaming_cache(device=current_device)

        subsampling_rate = self.encoder.embed.subsampling_rate if hasattr(
            self.encoder.embed, 'subsampling_rate') else 4

        encoder_static_chunk_frames = self.encoder.static_chunk_size if hasattr(
            self.encoder, 'static_chunk_size') and self.encoder.static_chunk_size > 0 else 16

        input_audio_chunk_frames = encoder_static_chunk_frames * subsampling_rate

        if chunk_size_ms is not None:
            input_audio_chunk_frames = int(chunk_size_ms / 10)

        num_input_audio_frames = audio_lens.item()
        current_input_offset_frames = 0

        min_input_audio_frames_for_conv = max(16, subsampling_rate * 4)

        if input_audio_chunk_frames < min_input_audio_frames_for_conv:
            if num_input_audio_frames >= min_input_audio_frames_for_conv:
                input_audio_chunk_frames = min_input_audio_frames_for_conv
            else:
                if num_input_audio_frames < 7:
                    print(f"Error: Input audio too short ({
                          num_input_audio_frames} frames) for conv layers. Skipping.")
                    return [[] for _ in range(audios.size(0))], None, None
                input_audio_chunk_frames = num_input_audio_frames

        while current_input_offset_frames < num_input_audio_frames:
            chunk_end_input_frames = min(
                current_input_offset_frames + input_audio_chunk_frames, num_input_audio_frames)
            current_chunk_audio = audios[:,
                                         current_input_offset_frames:chunk_end_input_frames, :]

            if current_chunk_audio.size(1) == 0:
                break

            if current_chunk_audio.size(1) < 7:
                current_input_offset_frames = chunk_end_input_frames
                continue

            encoder_output_offset = current_input_offset_frames // subsampling_rate
            required_enc_cache_size = encoder_output_offset

            self.streaming_beam_hypotheses, \
                self.streaming_att_cache, self.streaming_cnn_cache = self._decode_chunk_beam_search(
                    chunk_xs=current_chunk_audio,
                    offset=encoder_output_offset,
                    required_cache_size=required_enc_cache_size,
                    att_cache_in=self.streaming_att_cache,
                    cnn_cache_in=self.streaming_cnn_cache,
                    beam_hypotheses_in=self.streaming_beam_hypotheses,
                    beam_size=beam_size
                )

            current_input_offset_frames = chunk_end_input_frames

        if self.streaming_beam_hypotheses:
            best_hyp = max(self.streaming_beam_hypotheses,
                           key=lambda x: x.log_prob)
            return [best_hyp.tokens], None, None
        else:
            return [[]], None, None

    def process_single_chunk_beam_search(self, chunk_audio: torch.Tensor, chunk_len: torch.Tensor,
                                         beam_size: int = 4) -> Tuple[List[BeamHypothesis], None, None]:
        assert self.streaming, "Model is not in streaming mode for process_single_chunk_beam_search."
        assert chunk_audio.size(
            0) == 1, "Single chunk beam search currently supports batch size 1 only."

        current_device = chunk_audio.device

        if self.streaming_att_cache is None or self.streaming_cnn_cache is None:
            self.reset_streaming_cache(device=current_device)

        if chunk_audio.size(1) < 7:
            print(f"Warning: Chunk too small ({
                  chunk_audio.size(1)} frames), skipping")
            return self.streaming_beam_hypotheses or [], None, None

        if hasattr(self, '_global_encoder_offset'):
            encoder_output_offset = self._global_encoder_offset
        else:
            self._global_encoder_offset = 0
            encoder_output_offset = 0

        required_enc_cache_size = encoder_output_offset

        self.streaming_beam_hypotheses, \
            self.streaming_att_cache, self.streaming_cnn_cache = self._decode_chunk_beam_search(
                chunk_xs=chunk_audio,
                offset=encoder_output_offset,
                required_cache_size=required_enc_cache_size,
                att_cache_in=self.streaming_att_cache,
                cnn_cache_in=self.streaming_cnn_cache,
                beam_hypotheses_in=self.streaming_beam_hypotheses,
                beam_size=beam_size
            )

        subsampling_rate = self.encoder.embed.subsampling_rate if hasattr(
            self.encoder.embed, 'subsampling_rate') else 4
        estimated_encoder_frames = chunk_audio.size(1) // subsampling_rate
        self._global_encoder_offset += estimated_encoder_frames

        return self.streaming_beam_hypotheses, None, None

    def ctc_greedy_search(self, audios: torch.Tensor, audio_lens: torch.Tensor) -> List[List[int]]:
        if not self.ctc_head:
            return [[] for _ in range(audios.size(0))]

        current_device = audios.device
        audios, audio_lens = audios.to(
            current_device), audio_lens.to(current_device)

        encoder_out, encoder_mask = self.encoder(audios, audio_lens)

        ctc_log_probs = self.ctc_head.log_softmax(encoder_out)  # [B, T, V]
        ctc_preds_argmax = torch.argmax(ctc_log_probs, dim=2)  # [B, T]

        hyps = []
        for b in range(audios.size(0)):
            hyp_b = []
            prev_token = -1
            actual_len = encoder_mask[b].squeeze().sum().item()
            for t in range(actual_len):
                token = ctc_preds_argmax[b, t].item()
                if token != self.blank_id and token != prev_token:
                    hyp_b.append(token)
                prev_token = token
            hyps.append(hyp_b)
        return hyps
