# filepath: /home/celesteria/Course/SpeechProcess/exp3/online_rnnt_delay.py
import torch
import os
import time
import numpy as np
from tqdm import tqdm
from tokenizer.tokenizer import Tokenizer
from model.online_rnnt_model import OnlineRNNTModel
from data.dataloader import get_dataloader
from rnnt_common import Config
import argparse


def evaluate_rtf(model: OnlineRNNTModel, dataloader, device: torch.device, beam_size: int):
    """
    计算并打印模型在开发集上的实时因子 (RTF).
    """
    model.eval()

    chunk_size_frames = Config.static_chunk_size
    min_chunk_size = max(16, chunk_size_frames)
    frame_shift_s = 0.01  # 10ms frame shift

    all_rtfs_greedy = []
    all_rtfs_beam = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="RTF评估中"):
            audios = batch['audios'].to(device)
            audio_lens = batch['audio_lens'].to(device)
            batch_size = audios.size(0)

            for i in range(batch_size):
                audio_tensor = audios[i:i+1, :audio_lens[i]]

                # --- Greedy Search RTF for one audio ---
                model.reset_streaming_cache()
                current_offset = 0
                while current_offset < audio_tensor.shape[1]:
                    chunk_end = min(
                        current_offset + chunk_size_frames, audio_tensor.shape[1])
                    if audio_tensor.shape[1] - chunk_end < min_chunk_size and chunk_end < audio_tensor.shape[1]:
                        chunk_end = audio_tensor.shape[1]

                    current_chunk = audio_tensor[:,
                                                 current_offset:chunk_end, :]
                    if current_chunk.shape[1] == 0:
                        break

                    start_time = time.time()
                    model.process_single_chunk(
                        current_chunk, torch.tensor([current_chunk.shape[1]], device=device))
                    end_time = time.time()

                    processing_time = end_time - start_time
                    chunk_duration = current_chunk.shape[1] * frame_shift_s
                    rtf = processing_time / \
                        chunk_duration if chunk_duration > 0 else 0
                    all_rtfs_greedy.append(rtf)

                    current_offset = chunk_end
                    if chunk_end >= audio_tensor.shape[1]:
                        break

                # --- Beam Search RTF for one audio ---
                model.reset_streaming_cache()
                current_offset = 0
                while current_offset < audio_tensor.shape[1]:
                    chunk_end = min(
                        current_offset + chunk_size_frames, audio_tensor.shape[1])
                    if audio_tensor.shape[1] - chunk_end < min_chunk_size and chunk_end < audio_tensor.shape[1]:
                        chunk_end = audio_tensor.shape[1]

                    current_chunk = audio_tensor[:,
                                                 current_offset:chunk_end, :]
                    if current_chunk.shape[1] == 0:
                        break

                    start_time = time.time()
                    model.process_single_chunk_beam_search(
                        current_chunk, torch.tensor([current_chunk.shape[1]], device=device), beam_size=beam_size)
                    end_time = time.time()

                    processing_time = end_time - start_time
                    chunk_duration = current_chunk.shape[1] * frame_shift_s
                    rtf = processing_time / \
                        chunk_duration if chunk_duration > 0 else 0
                    all_rtfs_beam.append(rtf)

                    current_offset = chunk_end
                    if chunk_end >= audio_tensor.shape[1]:
                        break

    # --- 结果统计 ---
    print("\n" + "="*60)
    print("实时因子 (RTF) 统计结果")
    print("="*60)

    if all_rtfs_greedy:
        rtfs_greedy_np = np.array(all_rtfs_greedy)
        print("Greedy Search:")
        print(
            f"  - 平均RTF (Mean):         {np.mean(rtfs_greedy_np):.4f}")
        print(
            f"  - 50% (Median) RTF:       {np.percentile(rtfs_greedy_np, 50):.4f}")
        print(
            f"  - 80% RTF:                {np.percentile(rtfs_greedy_np, 80):.4f}")
        print(
            f"  - 90% RTF:                {np.percentile(rtfs_greedy_np, 90):.4f}")
        print(
            f"  - 95% RTF:                {np.percentile(rtfs_greedy_np, 95):.4f}")
        print(
            f"  - 最大RTF:                {np.max(rtfs_greedy_np):.4f}")
    else:
        print("Greedy Search: 未收集到RTF数据。")

    if all_rtfs_beam:
        rtfs_beam_np = np.array(all_rtfs_beam)
        print(f"\nBeam Search (beam_size={beam_size}):")
        print(
            f"  - 平均RTF (Mean):         {np.mean(rtfs_beam_np):.4f}")
        print(
            f"  - 50% (Median) RTF:       {np.percentile(rtfs_beam_np, 50):.4f}")
        print(
            f"  - 80% RTF:                {np.percentile(rtfs_beam_np, 80):.4f}")
        print(
            f"  - 90% RTF:                {np.percentile(rtfs_beam_np, 90):.4f}")
        print(
            f"  - 95% RTF:                {np.percentile(rtfs_beam_np, 95):.4f}")
        print(
            f"  - 最大RTF:                {np.max(rtfs_beam_np):.4f}")
    else:
        print(f"Beam Search (beam_size={beam_size}): 未收集到RTF数据。")


def main():
    parser = argparse.ArgumentParser(
        description="RNNT Streaming RTF Test on Dev Set")
    parser.add_argument("--model_path", type=str,
                        default="./online_model.pt", help="Path to model file")
    parser.add_argument("--beam_size", type=int, default=4,
                        help="Beam search size")

    args = parser.parse_args()

    model_path = args.model_path
    beam_size = args.beam_size

    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 未找到。")
        return

    tokenizer = Tokenizer()
    device = torch.device(Config.device)

    dataset_path = f"./dataset/split/{Config.eval_dataset}"
    dataloader = get_dataloader(
        f"{dataset_path}/wav.scp",
        f"{dataset_path}/pinyin",
        Config.batch_size,
        tokenizer,
        shuffle=False
    )

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
    model.eval()

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(
            f"模型加载完成，训练轮次: {checkpoint.get('epoch', -1)+1}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print("="*60)
    print(f"评估数据集: {Config.eval_dataset}")
    print(f"模型文件: {model_path}")
    print(f"Beam Search 大小: {beam_size}")
    print(
        f"流式配置: 块大小={Config.static_chunk_size}, 动态块={Config.use_dynamic_chunk}")
    print("="*60)

    evaluate_rtf(model, dataloader, device, beam_size)


if __name__ == "__main__":
    main()
