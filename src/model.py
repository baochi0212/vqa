
from transformers import ViTModel, MT5Model, AutoTokenizer
import torch.nn as nn
from dataset import EVJQA
from torch.utils import data
import torch
import warnings
warnings.filterwarnings('ignore')

class GenVQA(nn.Module):
    def __init__(self, vision_model, text_model, tokenizer, vocab_size=250112, image_embed_dim=768, text_embed_dim=512):
        super().__init__()
        '''
        vision model + encoder/decoder model
        '''
        self.tokenizer = tokenizer
        self.vision_model = vision_model
        self.vision_projection = nn.Linear(image_embed_dim, text_embed_dim)
        self.text_model = text_model
        self.fc = nn.Linear(text_embed_dim, vocab_size)

    def forward(self, image, question, answer, decoder_answer):
        '''
        convert input to mT5 __call__ inputs
        '''

        #visual embed
        image_embed = self.vision_model(image).last_hidden_state
        image_embed = self.vision_projection(image_embed)
        #preprend image to text:
        question_embed = self.text_model.encoder.embed_tokens(question)
        preprend_embed = torch.concat([image_embed, question_embed], dim=1)
        position_bias = None
        for block in self.text_model.encoder.block:
            preprend_embed, position_bias = block(preprend_embed, position_bias=position_bias)
        preprend_embed = self.text_model.encoder.final_layer_norm(preprend_embed)
        preprend_embed = self.text_model.encoder.dropout(preprend_embed)

        #decoder
        if self.training:
            decoder_outputs = self.text_model.decoder(input_ids=decoder_answer,
                                                      encoder_hidden_states=preprend_embed).last_hidden_state
            out = self.fc(decoder_outputs)
            loss = torch.nn.CrossEntropyLoss(ignore_index=0)(out.reshape(-1, out.shape[-1]), answer.reshape(-1))
            return loss
        else:
            decoder_outputs = self.text_model.decoder(input_ids=decoder_answer,
                                                      encoder_hidden_states=preprend_embed).last_hidden_state
            out = self.fc(decoder_outputs)
            loss = torch.nn.CrossEntropyLoss(ignore_index=0)(out.reshape(-1, out.shape[-1]), answer.reshape(-1))
            return out, loss #output and loss for tuning
        
    def generate(self, image, question, max_len=50):
        image, question = image.unsqueeze(0), question.unsqueeze(0)
        #visual embed
        image_embed = self.vision_model(image).last_hidden_state
        image_embed = self.vision_projection(image_embed)
        #preprend image to text:
        question_embed = self.text_model.encoder.embed_tokens(question)
        preprend_embed = torch.concat([image_embed, question_embed], dim=1)
        position_bias = None
        for block in self.text_model.encoder.block:
            preprend_embed, position_bias = block(preprend_embed, position_bias=position_bias)
        preprend_embed = self.text_model.encoder.final_layer_norm(preprend_embed)
        preprend_embed = self.text_model.encoder.dropout(preprend_embed)

        #decoder
        #decoder_inputs 1 x 1
        decoder_inputs = torch.tensor([[self.text_model.config.decoder_start_token_id]], dtype=torch.long).to(self.text_model.device)
        for i in range(max_len-1):
            decoder_outputs = self.text_model.decoder(input_ids=decoder_inputs,
                    encoder_hidden_states=preprend_embed).last_hidden_state
            decoder_outputs = self.fc(decoder_outputs)
            decoder_outputs = torch.argmax(decoder_outputs, -1)[0]
            # if self.tokenizer.decode(decoder_outputs[-1]) in ['<pad>', '</s>']:
            #     break
            
            # else:
            decoder_inputs = torch.cat([decoder_inputs, torch.tensor([[decoder_outputs[-1].item()]]).to(self.text_model.device)], dim=-1)
                
        return self.tokenizer.decode(decoder_outputs)

            
        
        


if __name__ == '__main__':
    #DATA
    image_dir = "/home/ubuntu/vqa/data/train-images"
    text_json_path = "/home/ubuntu/vqa/data/evjvqa_train.json"

    train_dataset = EVJQA(image_dir, text_json_path)
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=32,
                                       shuffle=True)
    sample = next(iter(train_dataloader))
    image, question, answer = sample
    
    
    
    #MODEL
    vit_path = "/home/ubuntu/vqa/huggingface_modules/models--google--vit-base-patch16-224-in21k/snapshots/7cbdb7ee3a6bcdf99dae654893f66519c480a0f8"
    t5_path = "/home/ubuntu/vqa/huggingface_modules/models--google--mt5-small/snapshots/38f23af8ec210eb6c376d40e9c56bd25a80f195d"
    vit_model = ViTModel.from_pretrained(vit_path)
    t5_model = MT5Model.from_pretrained(t5_path)
    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    model = GenVQA(vit_model, t5_model, tokenizer)
    # model.train()
    # print(model(*sample))
    model.eval()
    print(model.generate(image[0], question))




