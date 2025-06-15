from torch.utils.data import DataLoader, Dataset
from tokenizer.tokenizer import Tokenizer
import torch
import random
import os
from utils.utils import collate_with_PAD
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torchaudio as ta
import torch
import numpy as np
import librosa

def extract_audio_features(wav_file:str, n_fft=1024)->torch.Tensor:
    """
    从音频文件中提取FBank特征
    
    Args:
        wav_file: 音频文件路径
        n_fft: FFT窗口大小，默认为2048
        
    Returns:
        torch.Tensor: 提取的FBank特征，形状为 [time, n_mels]
    """
    waveform, sample_rate = torchaudio.load(wav_file)
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=80,
        hop_length=512,
        window_fn=torch.hamming_window,
        power=2.0
    )
    
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel_spec_db = mel_spec_db.squeeze(0).transpose(0, 1)
    
    return mel_spec_db


class BZNSYP(Dataset):
    def __init__(self, wav_file, text_file, tokenizer):
        self.tokenizer = tokenizer
        self.wav2path = {}
        self.wav2text = {}
        self.ids = []

        with open(wav_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    id = parts[0]
                    self.ids.append(id)
                    path = "./dataset/" + parts[1]
                    self.wav2path[id] = path
                else:
                    raise ValueError(f"Invalid line format: {line}")

        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    id = parts[0]
                    pinyin_list = parts[1].split(" ")
                    self.wav2text[id] = self.tokenizer(["<sos>"]+pinyin_list+["<eos>"])
                else:
                    raise ValueError(f"Invalid line format: {line}")
    
    def __len__(self):
        return len(self.wav2path)
    
    def __getitem__(self, index):
        id = list(self.wav2path.keys())[index]
        path = self.wav2path[id]
        text = self.wav2text[id]
        return id, extract_audio_features(path), text
    

def get_dataloader(wav_file, text_file, batch_size, tokenizer, shuffle=True, num_workers=0):
    dataset = BZNSYP(wav_file, text_file, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_with_PAD,
        num_workers=num_workers,
        pin_memory=True
    )