from data.dataloader import get_dataloader
from tokenizer.tokenizer import Tokenizer
from model.model import CTCModel
import torch
from utils.utils import to_device
import time
from tqdm import tqdm
import os
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 30
accum_steps = 1
grad_clip = 5

wandb.init(project="speech-recognition", 
           name="asr-ctc-model",
           config={
               "epochs": epochs,
               "device": device,
               "learning_rate": 0.0005,
               "batch_size": 16,
               "grad_clip": grad_clip,
               "accum_steps": accum_steps
           })

log_file = "./log.txt"
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write("训练日志\n")

tokenizer = Tokenizer()
model = CTCModel(80, 256, tokenizer.size(), tokenizer.blk_id()).to(device)
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr = 0.0005, betas=[0.9,0.98], eps= 1.0e-9, 
                                     weight_decay=1.0e-6, amsgrad= False )

train_dataloader = get_dataloader("./dataset/split/train/wav.scp", "./dataset/split/train/pinyin", 16 , tokenizer, shuffle=True)
test_dataloader = get_dataloader("./dataset/split/test/wav.scp", "./dataset/split/test/pinyin", 16 , tokenizer, shuffle=False)

print(f'Using device: {device}')

wandb.watch(model, log="all")

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
        predict, loss,_ = model(audios, audio_lens, texts, text_lens)
        loss = loss / accum_steps

        total_loss += loss.item()

        loss.backward()

        if (i+1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            optim.zero_grad()
        
        lr = optim.state_dict()['param_groups'][0]['lr']
        avg_loss = total_loss/(i+1)
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{lr:.6f}'
        })

        wandb.log({
            "train_loss": loss.item(),
            "learning_rate": lr,
            "step": epoch * len(train_dataloader) + i
        })

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

    wandb.log({
        "epoch": epoch,
        "train_epoch_loss": epoch_loss,
        "test_epoch_loss": test_loss
    })
    
    dict1 = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
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

wandb.finish()
