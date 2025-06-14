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

def pre_emphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    return np.append(x[0], x[1:] - alpha * x[:-1])

def framing(x: np.ndarray, sr: int, frame_lenth: float = 0.025, frame_gap: float = 0.010) -> np.ndarray:
    frame_len, frame_step = int(round(frame_lenth * sr)), int(round(frame_gap * sr))
    signal_len = len(x)

    if signal_len <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((signal_len - frame_len) / frame_step))

    pad_signal_length = (num_frames - 1) * frame_step + frame_len
    amount_to_pad = pad_signal_length - signal_len
    
    pad_signal = np.pad(x, (0, max(0, amount_to_pad)), mode='constant', constant_values=0)
    
    frame_indices_offset = np.arange(frame_len)
    frame_start_points = np.arange(num_frames) * frame_step
    
    indices = frame_start_points[:, np.newaxis] + frame_indices_offset[np.newaxis, :]
    
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def add_window(frame_sig: np.ndarray, sr: int, frame_len_s: float = 0.025) -> np.ndarray:
    window = np.hamming(int(round(frame_len_s * sr)))
    return frame_sig * window

def stft(frame_sig: np.ndarray, nfft: int = 512) -> tuple[np.ndarray, np.ndarray]:
    frame_spec = np.fft.rfft(frame_sig, n=nfft)
    frame_mag = np.abs(frame_spec)
    frame_pow = (frame_mag ** 2) / nfft
    return frame_mag, frame_pow

def get_filter_banks(sr, n_filters=40, nfft=512):
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((nfft + 1) * hz_points / sr).astype(int)
    filter_banks = np.zeros((n_filters, nfft // 2 + 1))
    fft_freqs = np.arange(nfft // 2 + 1)
    
    for i in range(n_filters):
        left, center, right = bins[i:i+3]

        left_mask = (left <= fft_freqs) & (fft_freqs < center)
        if center != left:
            filter_banks[i, left_mask] = (fft_freqs[left_mask] - left) / (center - left)

        right_mask = (center <= fft_freqs) & (fft_freqs < right)
        if right != center:
            filter_banks[i, right_mask] = (right - fft_freqs[right_mask]) / (right - center)
    
    return filter_banks

def get_fbank(frame_pow: np.ndarray, filter_banks: np.ndarray) -> np.ndarray:
    return np.dot(frame_pow, filter_banks.T)

def extract_audio_features(wav_file:str)->torch.Tensor:

    def calc_fbank(x: np.ndarray, sr: int = 16000, n_filters: int = 40, nfft: int = 512) -> np.ndarray:
        x = pre_emphasis(x)
        frames = framing(x, sr)
        frames = add_window(frames, sr)
        frame_mag, frame_pow = stft(frames, nfft)
        filter_banks = get_filter_banks(sr, n_filters, nfft)
        fbank = get_fbank(frame_pow, filter_banks)
        return fbank

    if not isinstance(wav_file, str):
        raise TypeError(f"Expected string for wav_file")

    y, sr = librosa.load(wav_file, sr=None)
    fbank = calc_fbank(y, sr=sr, n_filters=80, nfft=512)

    res = torch.from_numpy(fbank).float()

    if not isinstance(res, torch.Tensor):
        raise TypeError("Return value must be torch.Tensor")
    return res


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
    

def get_dataloader(wav_file, text_file, batch_size, tokenizer, shuffle=True):
    dataset = BZNSYP(wav_file, text_file, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_with_PAD
    )
    return dataloader