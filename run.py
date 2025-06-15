from data.dataloader import get_dataloader
from tokenizer.tokenizer import Tokenizer
from model.model import CTCModel
import torch
from utils.utils import to_device
import time
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 50
accum_steps = 1
grad_clip = 1.0

log_dir = "/root/tf-logs/speech-recognition-selfattention"
# log_dir = "./speech-recognition"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

log_file = "./log.txt"
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write("训练日志\n")

tokenizer = Tokenizer()
model = CTCModel(80, 256, tokenizer.size(), tokenizer.blk_id()).to(device)

initial_lr = 0.0001
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=initial_lr, betas=[0.9, 0.98], eps=1.0e-9,
                         weight_decay=1.0e-4, amsgrad=True)

scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)

warmup_steps = 4000
def get_lr_factor(step):
    if step < warmup_steps:
        return min(1.0, step / warmup_steps)
    return 1.0

train_dataloader = get_dataloader("./dataset/split/train/wav.scp", "./dataset/split/train/pinyin", 32, tokenizer, shuffle=True)
test_dataloader = get_dataloader("./dataset/split/test/wav.scp", "./dataset/split/test/pinyin", 32, tokenizer, shuffle=False)

print(f'Using device: {device}')

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
    print(f"\n第 {epoch+1}/{epochs} 轮训练")
    total_loss = 0.
    model.train()

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", ncols=100)
    
    for i, input in enumerate(progress_bar):
        input = to_device(input, device)
        audios = input['audios']
        audio_lens = input['audio_lens']
        texts = input['texts']
        text_lens = input['text_lens']
        
        if check_nan_inf(audios, "输入音频"):
            continue
            
        try:
            predict, loss, _ = model(audios, audio_lens, texts, text_lens)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"警告: 第 {i+1} 批次的损失为 NaN 或 Inf，跳过该批次")
                continue
                
            loss = loss / accum_steps
            total_loss += loss.item()
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
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{lr:.6f}'
        })

        global_step = epoch * len(train_dataloader) + i
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/learning_rate', lr, global_step)

        if (i+1) % (accum_steps*10) == 0:
            with open("./log.txt", 'a', encoding='utf-8') as f:
                f.write(f"{epoch}:{i+1}  {total_loss/(i+1)}  {lr}\n")
   
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
            predict, loss, _ = model(audios, audio_lens, texts, text_lens)
            test_loss += loss.item()
            test_progress.set_postfix({'test_loss': f'{test_loss/(i+1):.4f}'})
    
    test_loss = test_loss / len(test_dataloader)
    print(f"Epoch {epoch+1} done, : train loss: {epoch_loss:.4f}, test loss: {test_loss:.4f}")

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
