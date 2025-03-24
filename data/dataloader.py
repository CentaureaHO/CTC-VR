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

def extract_audio_features(wav_file:str)->torch.Tensor:
    if not isinstance(wav_file, str):
        raise TypeError(f"Expected string for wav_file")
    
    # TODO
    # 提取音频特征,并转化成torch.Tensor
    random_number = random.randint(100, 1000)
    res = torch.randn(random_number, 80)

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