import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm

import numpy as np
import pickle

import config
from data_utils.vocab import Vocab
from model.mcan import MCAN
from data_utils.vivqa_extracted_features import ViVQA, get_loader
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker

import os

def sample_example(metrics, net, dataset):
    print("TEST NUMBER:", len(dataset))
    with open(config.log_file, 'w') as f:
        for i in range(len(dataset)):
            v, q, a = [item.cuda().unsqueeze(0) for item in dataset[i]]
            out = net(v, q)
            prediction, answer, question = metrics.predict(out.cpu(), a.cpu(), q.cpu())
            for pred, a, q in zip(prediction, answer, question):
                if pred == a:
                    f.write(f"Question: {q}--- True Prediction: {prediction}--- Answer: {answer}\n")
                else:
                    f.write(f"Question: {q}--- Wrong Prediction: {prediction}--- Answer: {answer}\n")

        f.close()

if __name__ == '__main__':
    #metrics + test dataset
    metrics = Metrics()
    vocab = Vocab([config.json_train_path, config.json_test_path], 
                            specials=["<pad>", "<sos", "<eos>"])
    metrics.vocab = vocab
    test_dataset = ViVQA(config.json_test_path, vocab, config.test_image_dir)
    #log inference
    saved_info = torch.load(config.best_model_checkpoint + "/model_last.pth")
    net = MCAN(vocab, config.backbone, config.d_model, config.embedding_dim, config.image_patch_size, config.dff, config.nheads, 
                                config.nlayers, config.dropout).cuda()
    # torch.save({
    #     "weights": net.state_dict(),
    # }, './saved_models/temp.pth')
    # net.load_state_dict(torch.load("./saved_models/temp.pth")['weights'])
    net.load_state_dict(saved_info["weights"])
    net.eval()
    sample_example(metrics, net, test_dataset)