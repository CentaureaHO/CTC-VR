import torch
import os
import sys
from tokenizer.tokenizer import Tokenizer
from model.online_rnnt_model import OnlineRNNTModel
from data.dataloader import extract_audio_features
from rnnt_common import Config
import argparse


def decode_single_audio(audio_file: str, model_path: str = "./online_model.pt", beam_size: int = 4):
    print(f"正在加载音频文件: {audio_file}")

    if not os.path.exists(audio_file):
        print(f"错误: 音频文件 {audio_file} 不存在")
        return

    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return

    tokenizer = Tokenizer()
    print(f"Tokenizer加载完成，词汇表大小: {tokenizer.size()}")

    audio_features = extract_audio_features(audio_file)
    print(f"音频特征提取完成，形状: {audio_features.shape}")

    device = torch.device(Config.device)
    print(f"使用设备: {device}")

    print("正在加载模型...")
    model = OnlineRNNTModel(
        input_dim=80,
        hidden_dim=Config.hidden_dim,
        vocab_size=tokenizer.size(),
        blank_id=tokenizer.blk_id(),
        streaming=True,
        static_chunk_size=Config.static_chunk_size,
        use_dynamic_chunk=Config.use_dynamic_chunk,
        ctc_weight=Config.ctc_weight,
        predictor_layers=Config.predictor_layers,
        predictor_dropout=Config.predictor_dropout,
        ctc_dropout_rate=Config.ctc_dropout_rate,
        rnnt_loss_clamp=Config.rnnt_loss_clamp,
        ignore_id=Config.ignore_id
    ).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"模型加载完成，训练轮次: {checkpoint.get('epoch', -1)+1}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    audio_tensor = audio_features.unsqueeze(0).to(device)
    audio_len = torch.tensor([audio_features.shape[0]], device=device)

    print(f"输入音频张量形状: {audio_tensor.shape}")
    print(f"音频长度: {audio_len.item()} 帧")

    model.eval()

    print("\n" + "="*60)
    print("开始流式识别对比 (Greedy vs Beam Search)")
    print("="*60)

    chunk_size_frames = Config.static_chunk_size
    min_chunk_size = max(16, chunk_size_frames)

    print(f"流式配置: 每块 {chunk_size_frames} 帧 (最小 {min_chunk_size} 帧)")
    print(f"总音频长度: {audio_tensor.shape[1]} 帧")
    print(f"Beam Search 大小: {beam_size}")
    print("开始逐块处理...")

    with torch.no_grad():
        print("\n" + "-"*40)
        print("Greedy Search 流式处理")
        print("-"*40)
        
        model.reset_streaming_cache()
        greedy_tokens = []
        greedy_text = []
        current_offset = 0
        chunk_idx = 0

        while current_offset < audio_tensor.shape[1]:
            chunk_end = min(current_offset + chunk_size_frames, audio_tensor.shape[1])

            if audio_tensor.shape[1] - chunk_end < min_chunk_size and chunk_end < audio_tensor.shape[1]:
                chunk_end = audio_tensor.shape[1]

            current_chunk = audio_tensor[:, current_offset:chunk_end, :]
            chunk_len = torch.tensor([current_chunk.shape[1]], device=device)

            print(f"\n处理块 {chunk_idx+1}: 帧 {current_offset}:{chunk_end} (长度: {current_chunk.shape[1]})")

            try:
                chunk_tokens, _, _ = model.process_single_chunk(current_chunk, chunk_len)

                if chunk_tokens:
                    greedy_tokens.extend(chunk_tokens)
                    chunk_text = tokenizer.decode(chunk_tokens)
                    greedy_text += chunk_text
                    print(f"  Greedy输出: {' '.join(chunk_text)} (tokens: {chunk_tokens})")
                    print(f"  Greedy累积: {' '.join(greedy_text)}")
                else:
                    print(f"  Greedy输出: [无输出]")

            except Exception as e:
                print(f"  Greedy处理失败: {e}")

            current_offset = chunk_end
            chunk_idx += 1

            if chunk_end >= audio_tensor.shape[1]:
                break

        print("\n" + "-"*40)
        print("Beam Search 流式处理")
        print("-"*40)
        
        model.reset_streaming_cache()
        beam_tokens = []
        beam_text = []
        current_offset = 0
        chunk_idx = 0

        while current_offset < audio_tensor.shape[1]:
            chunk_end = min(current_offset + chunk_size_frames, audio_tensor.shape[1])

            if audio_tensor.shape[1] - chunk_end < min_chunk_size and chunk_end < audio_tensor.shape[1]:
                chunk_end = audio_tensor.shape[1]

            current_chunk = audio_tensor[:, current_offset:chunk_end, :]
            chunk_len = torch.tensor([current_chunk.shape[1]], device=device)

            print(f"\n处理块 {chunk_idx+1}: 帧 {current_offset}:{chunk_end} (长度: {current_chunk.shape[1]})")

            try:
                beam_hypotheses, _, _ = model.process_single_chunk_beam_search(
                    current_chunk, chunk_len, beam_size=beam_size)

                if beam_hypotheses:
                    best_hyp = max(beam_hypotheses, key=lambda x: x.log_prob)
 
                    new_tokens = best_hyp.tokens[len(beam_tokens):]
                    if new_tokens:
                        beam_tokens.extend(new_tokens)
                        chunk_text = tokenizer.decode(new_tokens)
                        beam_text += chunk_text
                        print(f"  Beam输出: {' '.join(chunk_text)} (tokens: {new_tokens})")
                        print(f"  Beam累积: {' '.join(beam_text)}")
                    else:
                        print(f"  Beam输出: [无新增输出]")
                        current_text = tokenizer.decode(best_hyp.tokens)
                        print(f"  Beam当前: {' '.join(current_text)}")
                    
                    print(f"  所有候选假设 ({len(beam_hypotheses)}个):")
                    sorted_hyps = sorted(beam_hypotheses, key=lambda x: x.log_prob, reverse=True)
                    
                    for i, hyp in enumerate(sorted_hyps):
                        hyp_text = tokenizer.decode(hyp.tokens)
                        print(f"    {i+1}. {' '.join(hyp_text)} (log_prob: {hyp.log_prob:.4f}, tokens: {hyp.tokens})")
                        
                else:
                    print(f"  Beam输出: [无输出]")

            except Exception as e:
                print(f"  Beam处理失败: {e}")
                import traceback
                traceback.print_exc()

            current_offset = chunk_end
            chunk_idx += 1

            if chunk_end >= audio_tensor.shape[1]:
                break

        print(f"\n" + "="*60)
        print("流式识别对比结果")
        print("="*60)
        print(f"Greedy Search 结果: {' '.join(greedy_text)}")
        print(f"Greedy Token序列: {greedy_tokens}")
        print(f"Beam Search 结果: {' '.join(beam_text)}")
        print(f"Beam Token序列: {beam_tokens}")
        print(f"处理了 {chunk_idx} 个音频块")

        if greedy_tokens == beam_tokens:
            print("Greedy和Beam Search的输出完全相同！")

    print("\n" + "="*60)
    print("识别完成")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="RNNT Streaming Decoder")
    parser.add_argument("--audio_id", type=str, default='008001', help="Audio ID to decode")
    parser.add_argument("--model_path", type=str, default="./online_model.pt", help="Path to model file")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam search size")
    
    args = parser.parse_args()

    audio_file = f'./dataset/Wave/{args.audio_id}.wav'
    model_path = args.model_path
    beam_size = args.beam_size

    print("="*60)
    print(f"音频文件: {audio_file}")
    print(f"模型文件: {model_path}")
    print(f"Beam Search 大小: {beam_size}")
    print(f"流式配置: 块大小={Config.static_chunk_size}, 动态块={Config.use_dynamic_chunk}")
    print("="*60)

    decode_single_audio(audio_file, model_path, beam_size)


if __name__ == "__main__":
    main()
