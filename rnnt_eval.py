from data.dataloader import get_dataloader
from tokenizer.tokenizer import Tokenizer
from model.rnnt_model import TransducerModel
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


def evaluate_model(dataloader, model, tokenizer, device='cpu', output_file=None, use_ctc=False):
    all_refs = []
    all_hyps = []

    if output_file:
        f_out = open(output_file, 'w', encoding='utf-8')

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
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

                if use_ctc:
                    # 使用CTC解码
                    hyps_batch = model.ctc_greedy_search(
                        single_audio, single_audio_len)
                    hyp = hyps_batch[0] if hyps_batch and len(
                        hyps_batch) > 0 else []
                else:
                    # 使用RNNT解码
                    hyps_batch, _, _ = model(single_audio, single_audio_len)
                    hyp = hyps_batch[0] if hyps_batch and len(
                        hyps_batch) > 0 else []

                ref = single_text[0, :single_text_len[0]].cpu().tolist()
                all_refs.append(ref)
                all_hyps.append(hyp)

                if output_file:
                    ref_text = tokenizer.decode(ref)
                    hyp_text = tokenizer.decode(hyp)
                    f_out.write(f"REF: {ref_text}\n")
                    f_out.write(f"HYP: {hyp_text}\n\n")

    if output_file:
        f_out.close()

    total_S = total_D = total_I = total_N = 0
    for ref, hyp in zip(all_refs, all_hyps):
        cer, S, D, I, N = calculate_cer(hyp, ref)
        total_S += S
        total_D += D
        total_I += I
        total_N += N

    final_cer = (total_S + total_D + total_I) / total_N if total_N > 0 else 1.0

    print(f"评估结果:")
    print(
        f"替换(S): {total_S}, 删除(D): {total_D}, 插入(I): {total_I}, 参考长度(N): {total_N}")
    print(f"CER: {final_cer:.4f} ({total_S+total_D+total_I}/{total_N})")

    print("\n样本对比 (前5个):")
    for i in range(min(5, len(all_refs))):
        print(f"参考: {tokenizer.decode(all_refs[i])}")
        print(f"预测: {tokenizer.decode(all_hyps[i])}")
        print()

    return final_cer


def main():
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
    model = TransducerModel(
        80, Config.hidden_dim,
        tokenizer_instance.size(),
        tokenizer_instance.blk_id(),
        ctc_weight=Config.ctc_weight
    ).to(current_device)

    if not os.path.exists(Config.eval_model_path):
        print(f"错误: 模型文件 {Config.eval_model_path} 未找到。")
        return

    checkpoint = torch.load(Config.eval_model_path,
                            map_location=current_device)
    model.load_state_dict(checkpoint['model'])
    print(
        f"模型已从 {Config.eval_model_path} 加载，训练轮次: {checkpoint.get('epoch', -1)+1}")

    decode_method = "CTC" if Config.use_ctc else "RNNT"
    print(f"使用 {decode_method} 解码方法")

    cer = evaluate_model(dataloader, model, tokenizer_instance,
                         current_device, Config.eval_output, Config.use_ctc)
    print(f"最终CER ({Config.eval_dataset}集, {decode_method}): {cer:.4f}")


if __name__ == "__main__":
    main()
