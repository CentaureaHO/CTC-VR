from data.dataloader import get_dataloader
from tokenizer.tokenizer import Tokenizer
from model.model import CTCModel
import torch
from utils.utils import to_device
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 30
accum_steps = 1
grad_clip = 5

tokenizer = Tokenizer()
model = CTCModel(80, 256, tokenizer.size(), tokenizer.blk_id()).to(device)
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr = 0.0005, betas=[0.9,0.98], eps= 1.0e-9, 
                                     weight_decay=1.0e-6, amsgrad= False )

train_dataloader = get_dataloader("./dataset/split/train/wav.scp", "./dataset/split/train/pinyin", 16 , tokenizer, shuffle=True)
test_dataloader = get_dataloader("./dataset/split/test/wav.scp", "./dataset/split/test/pinyin", 16 , tokenizer, shuffle=False)

for epoch in range(epochs):
    total_loss = 0.
    model.train()
    for i, input in enumerate(train_dataloader):
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

        if (i+1) % (accum_steps*10) == 0:
            with open("./log.txt", 'a', encoding='utf-8') as f:
                f.write(f"{epoch}:{i+1}  {total_loss/(i+1)}  {optim.state_dict()['param_groups'][0]['lr']}\n")
   
    dict1 = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        }
    torch.save(dict1, "./model1.pt")
