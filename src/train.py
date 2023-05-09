from transformers import ViTModel, MT5Model, AutoTokenizer
import torch.nn as nn
from dataset import EVJQA
from torch.utils import data
import torch
import warnings
from model import GenVQA
warnings.filterwarnings('ignore')


cuda_id = 1


def train(model, train_dataloader, optimizer, n_epochs=10, device='cuda'):
    model = model.to(device)
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for iter, batch in enumerate(train_dataloader):
            batch = [item.to(device) for item in batch]
            loss = model(*batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step() 
            if iter % 10 == 0:
                print(f"Iter: {iter}, Loss: {loss.item()}")
            if iter % 100 == 0: 
                model.eval()
                image, question, answer, _ = batch
                pred = model.generate(image[0], question[0])
                out, loss = model(*batch)
                print(f"Iter: {iter}, question: {tokenizer.decode(question[0], skip_special_tokens=True)}, answer: {tokenizer.decode(answer[0], skip_special_tokens=True)} Prediction: {pred}") 
                print("Eval output", tokenizer.decode(torch.argmax(out, -1)[0]))
                model.train()
                torch.save(model.cpu().state_dict(), './model.pt')
        
        print(f"Epoch {epoch}, Loss: {epoch_loss/len(train_dataloader)}")





if __name__ == '__main__':
    #DATA
    image_dir = "/home/ubuntu/vqa/data/train-images"
    text_json_path = "/home/ubuntu/vqa/data/evjvqa_train.json"

    train_dataset = EVJQA(image_dir, text_json_path)
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=8,
                                       shuffle=True,
                                       pin_memory=True)
    sample = next(iter(train_dataloader))
    image, question, answer, _ = sample
    
    
    
    #MODEL
    vit_path = "/home/ubuntu/vqa/huggingface_modules/models--google--vit-base-patch16-224-in21k/snapshots/7cbdb7ee3a6bcdf99dae654893f66519c480a0f8"
    t5_path = "/home/ubuntu/vqa/huggingface_modules/models--google--mt5-small/snapshots/38f23af8ec210eb6c376d40e9c56bd25a80f195d"
    vit_model = ViTModel.from_pretrained(vit_path)
    t5_model = MT5Model.from_pretrained(t5_path)
    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    model = GenVQA(vit_model, t5_model, tokenizer)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-6)
    train(model, train_dataloader, optimizer, device='cuda')