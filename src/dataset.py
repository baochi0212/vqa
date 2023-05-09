from torch.utils import data
import json
from glob import glob
from PIL import Image
import numpy as np
import os
from utils import *
# from .utils import *
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor


class EVJQA(data.Dataset):
    def __init__(self, image_dir, text_json_path):
        super().__init__()
        self.image_dir = image_dir
        self.text_dict = process_json(text_json_path)
        # self.processor = AutoProcessor.from_pretrained("/home/ubuntu/vqa/huggingface_modules/models--Salesforce--blip-vqa-capfilt-large/snapshots/c6af15bed424cf343aab3ff3bb31417ba272923a")
        # self.tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/vqa/huggingface_modules/models--Salesforce--blip-vqa-capfilt-large/snapshots/c6af15bed424cf343aab3ff3bb31417ba272923a")
        self.processor = AutoImageProcessor.from_pretrained("/home/ubuntu/vqa/huggingface_modules/models--google--vit-base-patch16-224-in21k/snapshots/7cbdb7ee3a6bcdf99dae654893f66519c480a0f8")
        self.tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/vqa/huggingface_modules/models--google--mt5-small/snapshots/38f23af8ec210eb6c376d40e9c56bd25a80f195d")
    def transform(self, image, question, answer):
        '''
        input to mT5
            - input_ids
            - attention_mask
            - labels
        ''' 
        inputs = {}
        image = self.processor(image,
                               return_tensors='pt').pixel_values[0]
        question_tokens = self.tokenizer(question,
                                return_tensors='pt',
                                padding='max_length',
                                max_length=64)
        question = question_tokens.input_ids[0]
        attention_mask = question_tokens.attention_mask[0]

        #decoder start token <pad>
        decoder_answer = self.tokenizer('<pad> ' + answer,
                                return_tensors='pt',
                                padding='max_length',
                                max_length=64).input_ids[0]
        answer = self.tokenizer(answer,
                                return_tensors='pt',
                                padding='max_length',
                                max_length=64).input_ids[0]
        # inputs['pixel_values'] = image
        # inputs['input_ids'] = question
        # inputs['attention_mask'] = attention_mask
        # inputs['labels'] = answer

        # # return image, question, answer
        # # inputs = self.processor(images=image, text=question, return_tensors="pt")
        # # labels = self.processor(text=answer, return_tensors="pt").input_ids
        # # inputs["labels"] = labels
        # return inputs

        return image, question, answer, decoder_answer
        

    def __getitem__(self, idx):
        image = Image.open(self.image_dir + '/' + self.text_dict[idx]['image_id'])
        question = self.text_dict[idx]['question']
        answer = self.text_dict[idx]['answer']

        #proces
        return self.transform(image, question, answer)

    def __len__(self):
        return len(self.text_dict)
    


if __name__ == '__main__':
    image_dir = "/home/ubuntu/vqa/data/train-images"
    text_json_path = "/home/ubuntu/vqa/data/evjvqa_train.json"

    dataset = EVJQA(image_dir, text_json_path)
    print(dataset[0][0].shape)


