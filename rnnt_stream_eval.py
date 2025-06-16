import os
import time
import torch
import torchaudio
import numpy as np
from tokenizer.tokenizer import Tokenizer
from model.rnnt import Transducer, RNNPredictor, TransducerJoint, StreamingEncoder
from typing import List, Tuple, Dict

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

CHUNK_SIZE_MS = 200                     # 每个音频块的毫秒数
SAMPLE_RATE = 16000                     # 采样率 (Hz)
FEATURE_DIM = 80                        # 特征维度
BLANK_ID = 0                            # blank标记ID
model_path = "./rnnt_model.pt"          # 模型路径
wav_path = "dataset/Wave/000020.wav"    # 测试文件

n_fft = 1024
hop_length = 512
n_mels = 80

class StreamBuffer:

    def __init__(self, chunk_size_ms: int, sample_rate: int, feature_params: Dict):
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.chunk_size_samples = int(chunk_size_ms * sample_rate / 1000)
        
        self.n_fft = feature_params.get("n_fft", 1024)
        self.n_mels = feature_params.get("n_mels", 80)
        self.hop_length = feature_params.get("hop_length", 512)

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            window_fn=torch.hamming_window,
            power=2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.audio_buffer = torch.zeros(1, 3 * self.chunk_size_samples)
        self.buffer_size = 3 * self.chunk_size_samples

        self.processed_samples = 0
    
    def add_audio_chunk(self, audio_chunk: torch.Tensor) -> None:
        current_size = min(self.buffer_size - audio_chunk.size(1), self.audio_buffer.size(1))
        self.audio_buffer[:, :current_size] = self.audio_buffer[:, -current_size:].clone()
        self.audio_buffer[:, current_size:current_size + audio_chunk.size(1)] = audio_chunk
        self.processed_samples += audio_chunk.size(1)
    
    def extract_features(self) -> torch.Tensor:
        valid_audio = self.audio_buffer[:, :self.buffer_size]
        
        mel_spec = self.mel_spec(valid_audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        features = mel_spec_db.squeeze(0).transpose(0, 1)
        
        frames_per_chunk = int(self.chunk_size_samples / self.hop_length)
        
        if features.size(0) > frames_per_chunk:
            return features[-frames_per_chunk:]
        else:
            return features

def load_model_and_tokenizer(model_path: str) -> Tuple[Transducer, Tokenizer]:
    tokenizer = Tokenizer()
    
    vocab_size = tokenizer.size()
    blank_id = tokenizer.blk_id()
    input_dim = 80
    enc_hidden_dim = 256
    enc_output_dim = 256
    pred_hidden_dim = 320
    pred_output_dim = 320
    joint_dim = 320
    rnn_layers = 2

    encoder = StreamingEncoder(
        input_size=input_dim,
        hidden_size=enc_hidden_dim,
        output_size=enc_output_dim,
        num_layers=rnn_layers,
        dropout=0.0,
        bidirectional=False
    )

    predictor = RNNPredictor(
        voca_size=vocab_size,
        embed_size=256,
        output_size=pred_output_dim,
        embed_dropout=0.0,
        hidden_size=pred_hidden_dim,
        num_layers=rnn_layers,
        rnn_type="lstm"
    )

    joint = TransducerJoint(
        vocab_size=vocab_size,
        enc_output_size=enc_output_dim,
        pred_output_size=pred_output_dim,
        join_dim=joint_dim,
        prejoin_linear=True,
        postjoin_linear=False,
        joint_mode='add'
    )

    model = Transducer(
        vocab_size=vocab_size,
        blank=blank_id,
        encoder=encoder,
        predictor=predictor,
        joint=joint,
        ignore_id=-1,
        transducer_weight=1.0,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"模型 {model_path} 加载完成")
    
    return model, tokenizer

def streaming_decode_rnnt(
    model: Transducer,
    wav_path: str,
    tokenizer: Tokenizer,
    chunk_size_ms: int = 200,
) -> str:
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    
    chunk_size_samples = int(chunk_size_ms * SAMPLE_RATE / 1000)
    feature_params = {
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length
    }
    stream_buffer = StreamBuffer(chunk_size_ms, SAMPLE_RATE, feature_params)
    encoder_cache = model.encoder.init_state(1, device=device)

    predictor_cache = model.forward_predictor_init_state(device=device)
    decoded_tokens = []
    current_token = tokenizer.blk_id()

    partial_results = []
    num_chunks = (waveform.size(1) + chunk_size_samples - 1) // chunk_size_samples
    
    print("\n开始流式解码...")
    print("-" * 60)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size_samples
        end_idx = min(start_idx + chunk_size_samples, waveform.size(1))
        current_chunk = waveform[:, start_idx:end_idx]
        
        if current_chunk.size(1) < chunk_size_samples:
            padded_chunk = torch.zeros(1, chunk_size_samples, device=current_chunk.device)
            padded_chunk[:, :current_chunk.size(1)] = current_chunk
            current_chunk = padded_chunk
        
        stream_buffer.add_audio_chunk(current_chunk)
        features = stream_buffer.extract_features()
        features = features.unsqueeze(0).to(device)
        enc_out, encoder_cache = model.encoder.forward_chunk(features, encoder_cache)
        
        for t in range(enc_out.size(1)):
            enc_out_t = enc_out[:, t:t+1, :]
            
            while True:
                pred_input = torch.tensor([[current_token]], dtype=torch.long, device=device)
                
                pred_out, predictor_cache = model.forward_predictor_step(pred_input, predictor_cache)
                logits = model.forward_joint_step(enc_out_t, pred_out)
                pred_token = torch.argmax(logits, dim=-1).item()
                
                if pred_token != tokenizer.blk_id():
                    decoded_tokens.append(pred_token)
                    current_token = pred_token
                    
                    if len(decoded_tokens) % 5 == 0 or t == enc_out.size(1) - 1:
                        partial_text = ''.join(tokenizer.decode(decoded_tokens))
                        if partial_text not in partial_results:
                            partial_results.append(partial_text)
                            progress = f"块 {i+1}/{num_chunks} | 已处理 {(i+1)/num_chunks*100:.1f}% | 当前识别: {partial_text}"
                            print(f"\r{progress}", end="", flush=True)
                else:
                    # blank
                    break
    
    print("\n" + "-" * 60)
    final_text = ''.join(tokenizer.decode(decoded_tokens))
    print(f"最终识别结果: {final_text}")
    
    return final_text

if __name__ == "__main__":
    print(f"加载模型中...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    print(f"\n开始处理音频: {wav_path}")
    start_time = time.time()
    result = streaming_decode_rnnt(model, wav_path, tokenizer, CHUNK_SIZE_MS)
    end_time = time.time()
    
    print(f"\n解码用时: {end_time - start_time:.2f} 秒")
