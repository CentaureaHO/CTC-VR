import torch
import os
import sys
from tokenizer.tokenizer import Tokenizer
from model.online_rnnt_model import OnlineRNNTModel
from data.dataloader import extract_audio_features
from rnnt_common import Config


def decode_single_audio(audio_file: str, model_path: str = "./online_model.pt"):
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
    print("开始流式识别...")
    print("="*60)

    chunk_size_frames = Config.static_chunk_size
    min_chunk_size = max(16, chunk_size_frames)

    print(f"流式配置: 每块 {chunk_size_frames} 帧 (最小 {min_chunk_size} 帧)")
    print(f"总音频长度: {audio_tensor.shape[1]} 帧")
    print("开始逐块处理...")

    with torch.no_grad():
        model.reset_streaming_cache()

        all_tokens = []
        current_text = []

        current_offset = 0
        chunk_idx = 0

        while current_offset < audio_tensor.shape[1]:
            chunk_end = min(current_offset + chunk_size_frames,
                            audio_tensor.shape[1])

            if audio_tensor.shape[1] - chunk_end < min_chunk_size and chunk_end < audio_tensor.shape[1]:
                chunk_end = audio_tensor.shape[1]

            current_chunk = audio_tensor[:, current_offset:chunk_end, :]
            chunk_len = torch.tensor([current_chunk.shape[1]], device=device)

            print(
                f"\n处理块 {chunk_idx+1}: 帧 {current_offset}:{chunk_end} (长度: {current_chunk.shape[1]})")

            try:
                chunk_tokens, _, _ = model.process_single_chunk(
                    current_chunk, chunk_len)

                if chunk_tokens:
                    all_tokens.extend(chunk_tokens)
                    chunk_text = tokenizer.decode(chunk_tokens)
                    current_text += chunk_text
                    print(
                        f"  块输出: {' '.join(chunk_text)} (tokens: {chunk_tokens})")
                    print(f"  累积结果: {' '.join(current_text)}")
                else:
                    print(f"  块输出: [无输出]")

            except Exception as e:
                print(f"  块处理失败: {e}")
                import traceback
                traceback.print_exc()

            current_offset = chunk_end
            chunk_idx += 1

            if chunk_end >= audio_tensor.shape[1]:
                break

        print(f"\n" + "="*60)
        print("流式识别完成")
        print("="*60)
        print(f"最终结果: {' '.join(current_text)}")
        print(f"最终Token序列: {all_tokens}")
        print(f"处理了 {chunk_idx} 个音频块")

    print("\n" + "="*60)
    print("识别完成")
    print("="*60)


def main():
    audio_file = "./dataset/Wave/000020.wav"
    model_path = "./online_model.pt"

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

    print("="*60)
    print(f"音频文件: {audio_file}")
    print(f"模型文件: {model_path}")
    print(
        f"流式配置: 块大小={Config.static_chunk_size}, 动态块={Config.use_dynamic_chunk}")
    print("="*60)

    decode_single_audio(audio_file, model_path)


if __name__ == "__main__":
    main()
