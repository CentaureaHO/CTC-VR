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
import argparse
from model.rnnt_model import TransducerModel

def main():
    parser = argparse.ArgumentParser(description="RNNT模型训练")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=0.0001, help="初始学习率")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--accum_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    
    parser.add_argument("--streaming", action="store_true", help="是否使用流式训练")
    parser.add_argument("--static_chunk_size", type=int, default=20, 
                        help="静态块大小，推荐16-32，越小延迟越低但可能影响精度")
    parser.add_argument("--use_dynamic_chunk", action="store_true", 
                        help="是否使用动态块训练，提高模型鲁棒性")
    parser.add_argument("--num_decoding_left_chunks", type=int, default=4, 
                        help="解码时使用的左侧块数，推荐4-8，影响历史信息利用")
    
    parser.add_argument("--ctc_weight", type=float, default=0.3,
                        help="CTC损失权重，RNNT权重为1-ctc_weight")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="训练设备")
    parser.add_argument("--log_dir", type=str, default="/root/tf-logs/speech-recognition-rnnt", 
                        help="TensorBoard日志目录")
    parser.add_argument("--save_dir", type=str, default="./models", 
                        help="模型保存目录")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    epochs = args.epochs
    accum_steps = args.accum_steps
    grad_clip = args.grad_clip

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    os.makedirs(args.save_dir, exist_ok=True)
    
    tokenizer = Tokenizer()
    vocab_size = tokenizer.size()
    blank_id = tokenizer.blk_id()
    
    model = TransducerModel(
        input_dim=80, 
        hidden_dim=args.hidden_dim, 
        vocab_size=vocab_size, 
        blank_id=blank_id,
        streaming=args.streaming,
        static_chunk_size=args.static_chunk_size if args.streaming else 0,
        use_dynamic_chunk=args.use_dynamic_chunk if args.streaming else False,
        ctc_weight=args.ctc_weight
    ).to(device)
    
    print(f"模型配置:")
    print(f"  - 使用流式训练: {args.streaming}")
    print(f"  - CTC权重: {args.ctc_weight}")
    print(f"  - RNNT权重: {1.0 - args.ctc_weight}")
    if args.streaming:
        print(f"  - 静态块大小: {args.static_chunk_size}")
        print(f"  - 使用动态块: {args.use_dynamic_chunk}")
        print(f"  - 左侧块数: {args.num_decoding_left_chunks}")
    
    initial_lr = args.lr
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=initial_lr, betas=[0.9, 0.98], eps=1.0e-9,
                             weight_decay=1.0e-4, amsgrad=True)
    
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)
    
    warmup_steps = 4000
    def get_lr_factor(step):
        if step < warmup_steps:
            return min(1.0, step / warmup_steps)
        return 1.0
    
    train_dataloader = get_dataloader("./dataset/split/train/wav.scp", "./dataset/split/train/pinyin", 
                                    args.batch_size, tokenizer, shuffle=True)
    test_dataloader = get_dataloader("./dataset/split/test/wav.scp", "./dataset/split/test/pinyin", 
                                   args.batch_size, tokenizer, shuffle=False)
    
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
                    print(f"参数 {name} 的梯度统计: min={grad.min().item()}, max={grad.max().item()}, mean={grad.mean().item()}")

    for epoch in range(epochs):
        print(f"\n第 {epoch+1}/{epochs} 轮RNNT训练")
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_rnnt_loss = 0.0
        model.train()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", ncols=120)
        
        for i, input in enumerate(progress_bar):
            input = to_device(input, device)
            audios = input['audios']
            audio_lens = input['audio_lens']
            texts = input['texts']
            text_lens = input['text_lens']
            
            if check_nan_inf(audios, "输入音频"):
                continue
            
            try:
                _, loss, loss_dict = model(audios, audio_lens, texts, text_lens)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"警告: 第 {i+1} 批次的损失为 NaN 或 Inf，跳过该批次")    
                    continue
                
                loss = loss / accum_steps
                total_loss += loss.item()
                
                # 累积详细损失信息
                if loss_dict:
                    if loss_dict.get('loss_ctc') is not None:
                        total_ctc_loss += loss_dict['loss_ctc'].item()
                    if loss_dict.get('loss_rnnt') is not None:
                        total_rnnt_loss += loss_dict['loss_rnnt'].item()
                
                loss.backward()

                if (i+1) % 50 == 0:
                    log_gradient_stats(model)

                if (i+1) % accum_steps == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            print(f"警告: {name} 的梯度包含 NaN 或 Inf 值，进行裁剪前")
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
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
            
            # 更新tqdm进度条显示
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
                    writer.add_scalar('train/loss_ctc', loss_dict['loss_ctc'].item(), global_step)
                if loss_dict.get('loss_rnnt') is not None:
                    writer.add_scalar('train/loss_rnnt', loss_dict['loss_rnnt'].item(), global_step)
        
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
                test_progress.set_postfix({'test_loss': f'{test_loss/((i+1)*audios.size(0)):.4f}'})
        
        test_loss = test_loss / len(test_dataloader.dataset)
        print(f"Epoch {epoch+1} 完成, RNNT训练损失: {epoch_loss:.4f}, 测试损失: {test_loss:.4f}")
        scheduler.step(test_loss)
        writer.add_scalar('epoch/train_loss', epoch_loss, epoch)
        writer.add_scalar('epoch/test_loss', test_loss, epoch)
        writer.add_scalar('epoch/learning_rate', optim.state_dict()['param_groups'][0]['lr'], epoch)

        dict1 = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "train_loss": epoch_loss,
            "test_loss": test_loss
        }
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model_path = f"./models/model_epoch_{epoch+1}.pt"
            os.makedirs("./models", exist_ok=True)
            torch.save(dict1, model_path)
        
        latest_model_path = "./model.pt"
        torch.save(dict1, latest_model_path)

    writer.close()

if __name__ == "__main__":
    main()
