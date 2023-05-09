from transformers import ViTModel, MT5Model, AutoTokenizer
import torch.nn as nn
from dataset import EVJQA
from torch.utils import data
import torch
import warnings
from model import GenVQA
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    #DATA
    image_dir = "/home/ubuntu/vqa/data/train-images"
    text_json_path = "/home/ubuntu/vqa/data/evjvqa_train.json"

    train_dataset = EVJQA(image_dir, text_json_path)
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=32,
                                       shuffle=True)
    sample = next(iter(train_dataloader))
    image, question, answer, decoder_answer = sample
    
    
    
    #MODEL
    vit_path = "/home/ubuntu/vqa/huggingface_modules/models--google--vit-base-patch16-224-in21k/snapshots/7cbdb7ee3a6bcdf99dae654893f66519c480a0f8"
    t5_path = "/home/ubuntu/vqa/huggingface_modules/models--google--mt5-small/snapshots/38f23af8ec210eb6c376d40e9c56bd25a80f195d"
    vit_model = ViTModel.from_pretrained(vit_path)
    t5_model = MT5Model.from_pretrained(t5_path)
    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    model = GenVQA(vit_model, t5_model, tokenizer).cpu()
    model.load_state_dict(torch.load('./model.pt'))
    model.eval()
    generate_tokens = model.generate(image[0], question[0])
    out, loss = model(*sample)
    print(loss)
    print("----generate-------")
    print(generate_tokens)
    print("-----masked foward--------")


    print(tokenizer.decode(torch.argmax(out, -1)[0]))
    print("------------answer-------------")
    print(tokenizer.decode(answer[0]))
    # print(tokenizer.decode(decoder_answer[0]))
