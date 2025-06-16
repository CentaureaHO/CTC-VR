import torch
import torch.nn as nn
from wenet.transducer.transducer import Transducer
from wenet.transducer.predictor import RNNPredictor as Predictor
from wenet.transducer.joint import TransducerJoint
from wenet.transformer.encoder import ConformerEncoder
from wenet.transformer.ctc import CTC

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