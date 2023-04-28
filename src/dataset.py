from torch.utils import data
import json
from glob import glob
from PIL import Image
import numpy as np
import os
from .utils import *
from transformers import AutoProcessor, AutoTokenizer


class EVJQA(data.Dataset):
    def __init__(self, image_dir, text_json_path):
        super().__init__()
        self.image_dir = image_dir
        self.text_dict = process_json(text_json_path)
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    def transform(self, image, question, answer):
        image = self.processor(image,
                               return_tensors='pt').pixel_values
        question = self.tokenizer(question,
                                return_tensors='pt',
                                padding='max_length',
                                max_length=50).input_ids
       
        answer = self.tokenizer(answer,
                                return_tensors='pt',
                                padding='max_length',
                                max_length=50).input_ids
        
        return image, question, answer

    def __getitem__(self, idx):
        image = Image.open(self.image_dir + '/' + self.text_dict[idx]['image_id'])
        question = self.text_dict[idx]['question']
        answer = self.text_dict[idx]['answer']

        #proces
        return self.transform(image, question, answer)

    def __len__(self):
        return len(self.text_dict)
    


if __name__ == '__main__':
    image_dir = "/home/ubuntu/VQA/vqa/data/EVJVQA/train-images"
    text_json_path = "/home/ubuntu/VQA/vqa/data/EVJVQA/evjvqa_train.json"

    dataset = EVJQA(image_dir, text_json_path)
    print([x.shape for x in dataset[0]])


