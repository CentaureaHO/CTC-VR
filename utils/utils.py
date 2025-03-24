import torch
import random
import os
import dataclasses
import numpy as np
def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """Change the device of object recursively"""
    if isinstance(data, dict):
        return {k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()}
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[to_device(v, device, dtype, non_blocking, copy) for v in dataclasses.astuple(data)]
        )
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[to_device(o, device, dtype, non_blocking, copy) for o in data])
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype, non_blocking, copy)
    else:
        return data
    
def collate_with_PAD(batch):
    ids = [data[0] for data in batch]
    audios = [data[1] for data in batch]
    texts = [data[2] for data in batch]

    audio_lens = [audio.size(0) for audio in audios]
    text_lens = [len(text) for text in texts]
    max_text_len = max(text_lens)
    max_audio_len = max(audio_lens)

    audio_features = []
    for audio in audios:
        l,d = audio.size()
        if l < max_audio_len:
            padding = torch.zeros((max_audio_len - l, d), dtype=torch.float32)
            padded_tensor = torch.cat([audio, padding], dim=0)
            audio_features.append(padded_tensor)
        else:
            audio_features.append(audio)
    audio_features = torch.stack(audio_features, dim=0)
    
    # text padding id
    pad_num = 0
    text_features = []
    for text in texts:
        pad_len = max_text_len - len(text)
        pad_text = text + [pad_num] * pad_len
        text_features.append(pad_text)
    text_features = torch.LongTensor(text_features)
    audio_lens = torch.IntTensor(audio_lens)
    text_lens = torch.IntTensor(text_lens)

    res = {
        "ids": ids,
        "audios": audio_features,
        "audio_lens": audio_lens,
        "texts": text_features,
        "text_lens": text_lens,
    }
    
    return res
