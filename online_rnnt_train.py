from data.dataloader import get_dataloader
from tokenizer.tokenizer import Tokenizer
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from utils.utils import to_device
from tqdm import tqdm
import os
import time
import numpy as np
from model.online_rnnt_model import OnlineRNNTModel
from rnnt_common import Config


def main():
    device = torch.device(Config.device)
    epochs = Config.epochs
    accum_steps = Config.accum_steps
    grad_clip = Config.grad_clip

    # 打印配置信息
    Config.log_dir = "/root/tf-logs/speech-recognition-rnnt-online"
    Config.print_config()
    print("使用在线RNNT模型进行流式训练")

    os.makedirs(Config.log_dir, exist_ok=True)
    writer = SummaryWriter(Config.log_dir)

    os.makedirs(Config.save_dir, exist_ok=True)

    tokenizer = Tokenizer()
    vocab_size = tokenizer.size()
    blank_id = tokenizer.blk_id()

    model = OnlineRNNTModel(
        input_dim=80,
        hidden_dim=Config.hidden_dim,
        vocab_size=vocab_size,
        blank_id=blank_id,
        streaming=Config.streaming,
        static_chunk_size=Config.static_chunk_size,
        use_dynamic_chunk=Config.use_dynamic_chunk,
        ctc_weight=Config.ctc_weight,
        predictor_layers=Config.predictor_layers,
        predictor_dropout=Config.predictor_dropout,
        ctc_dropout_rate=Config.ctc_dropout_rate,
        rnnt_loss_clamp=Config.rnnt_loss_clamp,
        ignore_id=Config.ignore_id
    ).to(device)

    print(f"在线RNNT模型配置:")
    print(f"  - 流式处理: {Config.streaming}")
    print(f"  - 因果卷积: 已启用")
    print(f"  - 块大小: {Config.static_chunk_size}")
    print(f"  - 动态块: {Config.use_dynamic_chunk}")
    print(f"  - CTC权重: {Config.ctc_weight}")

    initial_lr = Config.lr
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=initial_lr, betas=[0.9, 0.98], eps=1.0e-9,
                             weight_decay=1.0e-4, amsgrad=True)

    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)

    warmup_steps = 4000

    def get_lr_factor(step):
        if step < warmup_steps:
            return min(1.0, step / warmup_steps)
        return 1.0

    train_dataloader = get_dataloader(Config.train_wav_scp, Config.train_text,
                                      Config.batch_size, tokenizer, shuffle=True)
    test_dataloader = get_dataloader(Config.test_wav_scp, Config.test_text,
                                     Config.batch_size, tokenizer, shuffle=False)

    print(f'使用设备: {device}')

    def check_nan_inf(tensor, name="tensor"):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"警告: {name} 包含 NaN 或 Inf 值")
            return True
        return False

    def log_gradient_stats(model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                if check_nan_inf(grad, f"梯度 {name}"):
                    print(
                        f"参数 {name} 的梯度统计: min={grad.min().item()}, max={grad.max().item()}, mean={grad.mean().item()}")

    for epoch in range(epochs):
        print(f"\n第 {epoch+1}/{epochs} 轮在线RNNT训练")
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_rnnt_loss = 0.0
        model.train()

        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}", ncols=120)

        for i, input in enumerate(progress_bar):
            input = to_device(input, device)
            audios = input['audios']
            audio_lens = input['audio_lens']
            texts = input['texts']
            text_lens = input['text_lens']

            if check_nan_inf(audios, "输入音频"):
                continue

            try:
                _, loss, loss_dict = model(
                    audios, audio_lens, texts, text_lens)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"警告: 第 {i+1} 批次的损失为 NaN 或 Inf，跳过该批次")
                    continue

                loss = loss / accum_steps
                total_loss += loss.item()

                if loss_dict:
                    if loss_dict.get('loss_ctc') is not None:
                        total_ctc_loss += loss_dict['loss_ctc']
                    if loss_dict.get('loss_rnnt') is not None:
                        total_rnnt_loss += loss_dict['loss_rnnt']

                loss.backward()

                if (i+1) % 50 == 0:
                    log_gradient_stats(model)

                if (i+1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip)

                    step = epoch * len(train_dataloader) + i
                    lr_factor = get_lr_factor(step)

                    for param_group in optim.param_groups:
                        param_group['lr'] = initial_lr * lr_factor

                    optim.step()
                    optim.zero_grad()

            except RuntimeError as e:
                print(f"运行时错误: {e}")
                continue

            lr = optim.state_dict()['param_groups'][0]['lr']
            avg_loss = total_loss/(i+1)
            avg_ctc_loss = total_ctc_loss/(i+1) if total_ctc_loss > 0 else 0
            avg_rnnt_loss = total_rnnt_loss/(i+1) if total_rnnt_loss > 0 else 0

            postfix_dict = {
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.6f}'
            }
            if avg_ctc_loss > 0:
                postfix_dict['ctc'] = f'{avg_ctc_loss:.4f}'
            if avg_rnnt_loss > 0:
                postfix_dict['rnnt'] = f'{avg_rnnt_loss:.4f}'

            progress_bar.set_postfix(postfix_dict)

            global_step = epoch * len(train_dataloader) + i

            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', lr, global_step)

            if loss_dict:
                if loss_dict.get('loss_ctc') is not None:
                    writer.add_scalar('train/loss_ctc',
                                      loss_dict['loss_ctc'], global_step)
                if loss_dict.get('loss_rnnt') is not None:
                    writer.add_scalar('train/loss_rnnt',
                                      loss_dict['loss_rnnt'], global_step)

        epoch_loss = total_loss / len(train_dataloader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            test_progress = tqdm(test_dataloader, desc=f"Testing", ncols=100)
            for i, input in enumerate(test_progress):
                input = to_device(input, device)
                audios = input['audios']
                audio_lens = input['audio_lens']
                texts = input['texts']
                text_lens = input['text_lens']

                _, loss, _ = model(audios, audio_lens, texts, text_lens)
                test_loss += loss.item() * audios.size(0)
                test_progress.set_postfix(
                    {'test_loss': f'{test_loss/((i+1)*audios.size(0)):.4f}'})

        test_loss = test_loss / len(test_dataloader.dataset)
        print(
            f"Epoch {epoch+1} 完成, 在线RNNT训练损失: {epoch_loss:.4f}, 测试损失: {test_loss:.4f}")

        scheduler.step(test_loss)
        writer.add_scalar('epoch/train_loss', epoch_loss, epoch)
        writer.add_scalar('epoch/test_loss', test_loss, epoch)
        writer.add_scalar('epoch/learning_rate',
                          optim.state_dict()['param_groups'][0]['lr'], epoch)

        dict1 = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "train_loss": epoch_loss,
            "test_loss": test_loss
        }

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model_path = f"{Config.save_dir}/online_model_epoch_{epoch+1}.pt"
            torch.save(dict1, model_path)

        latest_model_path = "./online_model.pt"
        torch.save(dict1, latest_model_path)

    writer.close()


if __name__ == "__main__":
    main()
