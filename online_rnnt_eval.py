from data.dataloader import get_dataloader
from tokenizer.tokenizer import Tokenizer
from model.online_rnnt_model import OnlineRNNTModel
import torch
from utils.utils import to_device
from tqdm import tqdm
import os
from rnnt_common import Config


def calculate_cer(pre_tokens: list, gt_tokens: list) -> tuple:
    m, n = len(pre_tokens), len(gt_tokens)

    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if pre_tokens[i-1] == gt_tokens[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    i, j = m, n
    S = D = I = 0

    while i > 0 and j > 0:
        if pre_tokens[i-1] == gt_tokens[j-1]:
            i -= 1
            j -= 1
        else:
            if dp[i][j] == dp[i-1][j-1] + 1:
                S += 1
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                D += 1
                i -= 1
            else:
                I += 1
                j -= 1

    D += i
    I += j

    N = len(gt_tokens)
    cer = (S + D + I) / N if N != 0 else 0.0
    return cer, S, D, I, N


def evaluate_streaming(dataloader, model, tokenizer, device='cpu',
                       output_file=None, beam_size=4):
    """评估流式beam search RNNT模型"""
    all_refs = []
    all_hyps_greedy = []
    all_hyps_beam = []

    if output_file:
        f_out = open(output_file, 'w', encoding='utf-8')

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="流式Beam Search评估中"):
            batch = to_device(batch, device)
            audios = batch['audios']
            audio_lens = batch['audio_lens']
            texts = batch['texts']
            text_lens = batch['text_lens']

            batch_size = audios.size(0)

            for j in range(batch_size):
                single_audio = audios[j:j+1]
                single_audio_len = audio_lens[j:j+1]
                single_text = texts[j:j+1]
                single_text_len = text_lens[j:j+1]

                model.reset_streaming_cache()
                hyps_greedy, _, _ = model.streaming_inference(
                    single_audio, single_audio_len)
                hyp_greedy = hyps_greedy[0] if hyps_greedy and len(
                    hyps_greedy) > 0 else []

                model.reset_streaming_cache()
                hyps_beam, _, _ = model.streaming_beam_search(
                    single_audio, single_audio_len, beam_size=beam_size)
                hyp_beam = hyps_beam[0] if hyps_beam and len(
                    hyps_beam) > 0 else []

                ref = single_text[0, :single_text_len[0]].cpu().tolist()
                all_refs.append(ref)
                all_hyps_greedy.append(hyp_greedy)
                all_hyps_beam.append(hyp_beam)

                if output_file:
                    ref_text = tokenizer.decode(ref)
                    hyp_greedy_text = tokenizer.decode(hyp_greedy)
                    hyp_beam_text = tokenizer.decode(hyp_beam)
                    f_out.write(f"REF:    {ref_text}\n")
                    f_out.write(f"GREEDY: {hyp_greedy_text}\n")
                    f_out.write(f"BEAM:   {hyp_beam_text}\n\n")

    if output_file:
        f_out.close()

    total_S = total_D = total_I = total_N = 0
    for ref, hyp in zip(all_refs, all_hyps_greedy):
        cer, S, D, I, N = calculate_cer(hyp, ref)
        total_S += S
        total_D += D
        total_I += I
        total_N += N

    greedy_cer = (total_S + total_D + total_I) / \
        total_N if total_N > 0 else 1.0

    total_S = total_D = total_I = total_N = 0
    for ref, hyp in zip(all_refs, all_hyps_beam):
        cer, S, D, I, N = calculate_cer(hyp, ref)
        total_S += S
        total_D += D
        total_I += I
        total_N += N

    beam_cer = (total_S + total_D + total_I) / total_N if total_N > 0 else 1.0

    print(f"流式Greedy Search 评估结果:")
    print(f"CER: {greedy_cer:.4f}")

    print(f"\n流式Beam Search (beam_size={beam_size}) 评估结果:")
    print(f"CER: {beam_cer:.4f}")

    print(f"\nCER改进: {
          greedy_cer - beam_cer:.4f} ({'提升' if beam_cer < greedy_cer else '下降'})")

    print("\n样本对比 (前5个):")
    for i in range(min(5, len(all_refs))):
        print(f"参考:   {tokenizer.decode(all_refs[i])}")
        print(f"Greedy: {tokenizer.decode(all_hyps_greedy[i])}")
        print(f"Beam:   {tokenizer.decode(all_hyps_beam[i])}")
        print()

    return greedy_cer, beam_cer


def main():
    beam_size = 4
    if len(os.sys.argv) > 1:
        beam_size = int(os.sys.argv[1])

    tokenizer_instance = Tokenizer()

    dataset_path = f"./dataset/split/{Config.eval_dataset}"
    dataloader = get_dataloader(
        f"{dataset_path}/wav.scp",
        f"{dataset_path}/pinyin",
        Config.batch_size,
        tokenizer_instance,
        shuffle=False
    )

    current_device = torch.device(Config.device)

    model = OnlineRNNTModel(
        input_dim=80,
        hidden_dim=Config.hidden_dim,
        vocab_size=tokenizer_instance.size(),
        blank_id=tokenizer_instance.blk_id(),
        streaming=Config.streaming,
        static_chunk_size=Config.static_chunk_size,
        use_dynamic_chunk=Config.use_dynamic_chunk,
        ctc_weight=Config.ctc_weight,
        predictor_layers=Config.predictor_layers,
        predictor_dropout=Config.predictor_dropout,
        ctc_dropout_rate=Config.ctc_dropout_rate,
        rnnt_loss_clamp=Config.rnnt_loss_clamp,
        ignore_id=Config.ignore_id
    ).to(current_device)

    model_path = "./online_model.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 未找到。")
        return

    checkpoint = torch.load(model_path, map_location=current_device)
    model.load_state_dict(checkpoint['model'])
    print(f"在线模型已从 {model_path} 加载，训练轮次: {checkpoint.get('epoch', -1)+1}")

    print("=" * 60)
    print(f"流式Beam Search RNNT模型评估 (beam_size={beam_size})")
    print("=" * 60)

    output_file = Config.eval_output
    if output_file:
        output_file = output_file.replace(
            '.txt', f'_beam_search_{beam_size}.txt')

    greedy_cer, beam_cer = evaluate_streaming(
        dataloader, model, tokenizer_instance, current_device,
        output_file, beam_size=beam_size
    )

    print("\n" + "="*60)
    print("评估总结:")
    print(f"流式Greedy Search CER: {greedy_cer:.4f}")
    print(f"流式Beam Search CER:   {beam_cer:.4f}")
    print(f"Beam Search相对改进:   {
          ((greedy_cer - beam_cer) / greedy_cer * 100):.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
