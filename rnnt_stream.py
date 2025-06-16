import torch
import os
import argparse
from model.rnnt_model import TransducerModel
from tokenizer.tokenizer import Tokenizer
from data.dataloader import extract_audio_features
import torchaudio
import numpy as np
from tqdm import tqdm
import difflib

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    tokenizer = Tokenizer()
    vocab_size = tokenizer.size()
    blank_id = tokenizer.blk_id()
    
    model = TransducerModel(
        input_dim=80, 
        hidden_dim=256, 
        vocab_size=vocab_size, 
        blank_id=blank_id,
        streaming=True,
        static_chunk_size=0,
        use_dynamic_chunk=True
    ).to(device)
    
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model, tokenizer

def simulate_streaming(audio_features, chunk_size=16, context=5):
    total_frames = audio_features.size(0)
    num_chunks = (total_frames + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_frame = i * chunk_size
        end_frame = min(start_frame + chunk_size, total_frames)
        chunk = audio_features[start_frame:end_frame].unsqueeze(0)
        
        yield chunk, (i == num_chunks-1)

def offline_recognition(model, audio_features, device):
    with torch.no_grad():
        features = audio_features.unsqueeze(0).to(device)
        feature_lens = torch.tensor([features.size(1)], device=device)

        hyps = model.transducer.greedy_search(
            features,
            feature_lens,
            decoding_chunk_size=-1,
            simulate_streaming=False
        )
        
        return hyps[0] if len(hyps) > 0 else []

def compare_results(stream_result, offline_result):
    stream_text = ' '.join(stream_result)
    offline_text = ' '.join(offline_result)
    
    if stream_text == offline_text:
        return "完全一致"
    
    differ = difflib.Differ()
    diff = list(differ.compare(stream_text.split(), offline_text.split()))
    
    stream_words = stream_text.split()
    offline_words = offline_text.split()
    
    if len(stream_words) == 0 and len(offline_words) == 0:
        return "两者均为空"
    
    total_words = max(len(stream_words), len(offline_words))
    
    common = sum(1 for w in stream_words if w in offline_words)
    similarity = common / total_words if total_words > 0 else 0
    return f"相似度: {similarity:.2%}\n差异详情: {' '.join(d for d in diff if d.startswith('+ ') or d.startswith('- '))}"

def main():
    parser = argparse.ArgumentParser(description="RNNT流式与离线识别对比")
    parser.add_argument("--model", type=str, default="./model.pt", 
                        help="模型路径")
    parser.add_argument("--audio", type=str, default="dataset/Wave/000020.wav", 
                        help="音频文件路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="设备")
    parser.add_argument("--chunk_size", type=int, default=16, 
                        help="流式处理的块大小")
    parser.add_argument("--num_left_chunks", type=int, default=5,
                        help="解码时使用的左侧块数")
    
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"加载模型: {args.model}")
    model, tokenizer = load_model(args.model, device)

    print(f"处理音频: {args.audio}")
    audio_features = extract_audio_features(args.audio)

    print("执行离线识别...")
    offline_start_time = time.time()
    offline_hyps = offline_recognition(model, audio_features, device)
    offline_tokens = [tokenizer.id2token[id] for id in offline_hyps]
    offline_result = ' '.join(offline_tokens)
    offline_time = time.time() - offline_start_time
    print(f"离线识别结果: {offline_result}")
    print(f"离线识别耗时: {offline_time:.4f}秒")
    print("-" * 50)

    print("执行流式识别...")
    stream_start_time = time.time()
    
    streaming_feature = []
    current_result = []
    
    for chunk, is_final in tqdm(simulate_streaming(audio_features, args.chunk_size)):
        chunk = chunk.to(device)
        
        streaming_feature.append(chunk)
        current_feature = torch.cat(streaming_feature, dim=1)
        current_feature_len = torch.tensor([current_feature.size(1)], device=device)
        
        with torch.no_grad():
            hyps = model.transducer.greedy_search(
                current_feature, 
                current_feature_len,
                decoding_chunk_size=args.chunk_size,
                num_decoding_left_chunks=args.num_left_chunks,
                simulate_streaming=True
            )
            
            if len(hyps) > 0:
                partial_result = [tokenizer.id2token[id] for id in hyps[0]]
                current_result = partial_result

                if not is_final:
                    print(f"部分识别结果: {' '.join(partial_result)}", end="\r")
    
    stream_time = time.time() - stream_start_time
    
    if len(current_result) > 0:
        stream_result = ' '.join(current_result)
        print(f"\n流式识别结果: {stream_result}")
    else:
        stream_result = ""
        print("\n未能识别出任何结果")
    
    print(f"流式识别耗时: {stream_time:.4f}秒")
    print("-" * 50)
    
    print("识别结果对比:")
    print(f"离线识别: {offline_result}")
    print(f"流式识别: {stream_result}")
    difference = compare_results(current_result, offline_tokens)
    print(f"差异分析: {difference}")
    print(f"速度对比: 流式耗时 {stream_time:.4f}秒, 离线耗时 {offline_time:.4f}秒")

if __name__ == "__main__":
    import time
    main()
