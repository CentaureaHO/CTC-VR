import torch
import torch.nn as nn
from wenet.transducer.transducer import Transducer
from wenet.transducer.predictor import RNNPredictor as Predictor
from wenet.transducer.joint import TransducerJoint
from wenet.transformer.encoder import ConformerEncoder

class TransducerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, blank_id):
        super().__init__()
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
            pos_enc_layer_type="rel_pos"
        )
        
        self.predictor = Predictor(
            voca_size=vocab_size,
            embed_size=hidden_dim,
            output_size=hidden_dim,
            embed_dropout=0.1,
            hidden_size=hidden_dim,
            dropout=0.1,
            num_layers=1
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

        self.transducer = Transducer(
            vocab_size=vocab_size,
            blank=blank_id,
            encoder=self.encoder,
            predictor=self.predictor,
            joint=self.joint,
            ctc=None,
            ctc_weight=0.0,
            ignore_id=-1,
            transducer_weight=1.0
        )
        
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
            return None, loss, None
        else:
            if texts is not None:
                outputs = self.transducer(batch, current_device)
                loss = outputs['loss']
                
                return None, loss, None
            else:
                hyps = self.transducer.greedy_search(audios, audio_lens)
                scores = None 
                return hyps, scores, None
