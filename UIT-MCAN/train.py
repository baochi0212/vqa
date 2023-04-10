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


total_iterations = 0
metrics = Metrics()

def run(net, loader, fold_idx, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
        loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params)) 
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        acc_tracker = tracker.track('{}_accuracy'.format(prefix), tracker_class(**tracker_params))
        pre_tracker = tracker.track('{}_precision'.format(prefix), tracker_class(**tracker_params))
        rec_tracker = tracker.track('{}_recall'.format(prefix), tracker_class(**tracker_params))
        f1_tracker = tracker.track('{}_F1'.format(prefix), tracker_class(**tracker_params))


    tq = tqdm(loader, desc='Epoch {:03d}'.format(epoch), ncols=0)
    loss_objective = nn.CrossEntropyLoss(label_smoothing=0.2).cuda()
    for v, q, a in tq:
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()
        out = net(v, q)
        scores = metrics.get_scores(out.cpu(), a.cpu())
        if train:
            optimizer.zero_grad()
            loss = loss_objective(out, a)
            loss_tracker.append(loss.item())
            loss.backward()
            optimizer.step()
        else:
            loss = np.array(0)
            acc_tracker.append(scores["accuracy"])
            pre_tracker.append(scores["precision"])
            rec_tracker.append(scores["recall"])
            f1_tracker.append(scores["F1"])
        fmt = '{:.4f}'.format
        if train:
            tq.set_postfix(loss=fmt(loss.item()))
        else:
            tq.set_postfix(accuracy=fmt(acc_tracker.mean.value), 
                            precision=fmt(pre_tracker.mean.value), recall=fmt(rec_tracker.mean.value), f1=fmt(f1_tracker.mean.value))
        tq.update()

        # if train:
        #     torch.save({
        #         "fold": loaders.index(loader),
        #         "epoch": epoch,
        #         "loss": loss_tracker.mean.value,
        #         "weights": net.state_dict()
        #     }, os.path.join(config.tmp_model_checkpoint, "last_model.pth"))

    if not train:
        return {
            "accuracy": acc_tracker.mean.value,
            "precision": pre_tracker.mean.value,
            "recall": rec_tracker.mean.value,
            "F1": f1_tracker.mean.value
        }
    else:
        return loss_tracker.mean.value


def main():

    cudnn.benchmark = True

    # if os.path.isfile(os.path.join(config.model_checkpoint, "vocab.pkl")):
    #     vocab = pickle.load(open(os.path.join(config.model_checkpoint, "vocab.pkl"), "rb"))
    # else:
    #     vocab = Vocab([config.json_train_path, config.json_test_path], 
    #                         specials=["<pad>", "<sos", "<eos>"], vectors=config.word_embedding)
    #     pickle.dump(vocab, open(os.path.join(config.model_checkpoint, "vocab.pkl"), "wb"))
    vocab = Vocab([config.json_train_path, config.json_test_path], 
                            specials=["<pad>", "<sos", "<eos>"], vectors=config.word_embedding)
    train_image_dir = config.train_image_dir
    test_image_dir = config.test_image_dir
    train_dataset = ViVQA(config.json_train_path, vocab, train_image_dir)
    metrics.vocab = vocab
    # for i in range(len(train_dataset)):
    #     train_dataset[i]
    test_dataset = ViVQA(config.json_test_path, vocab, test_image_dir)
    train_loader, test_loader = get_loader(train_dataset, test_dataset, k_fold=False)
    # if os.path.isfile(os.path.join(config.model_checkpoint, "folds.pkl")):
    #     folds, test_fold = pickle.load(open(os.path.join(config.model_checkpoint, "folds.pkl"), "rb"))
    # else:
    #     folds, test_fold = get_loader(train_dataset, config, test_dataset)
    #     pickle.dump((folds, test_fold), open(os.path.join(config.model_checkpoint, "folds.pkl"), "wb"))

    if config.start_from:
        saved_info = torch.load(config.start_from)
        from_epoch = saved_info["epoch"]
        from_fold = saved_info["fold"] + 1
        loss = saved_info["loss"]
        net = nn.DataParallel(MCAN(vocab, config.backbone, config.d_model, config.embedding_dim, config.image_patch_size, config.dff, config.nheads, 
                                    config.nlayers, config.dropout)).cuda()
        net.load_state_dict(saved_info["weights"])
    else:
        from_epoch = 0
        from_fold = 0
        net = None
        loss = None

    if net is None:
        net = nn.DataParallel(MCAN(vocab, config.backbone, config.d_model, config.embedding_dim, config.image_patch_size, config.dff, config.nheads, 
                                    config.nlayers, config.dropout)).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)

    tracker = Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    max_f1 = 0 # for saving the best model
    for e in range(from_epoch, config.epochs):
        loss = run(net, train_loader, from_fold, optimizer, tracker, train=True, prefix='Training', epoch=e)
        
        if loss:
            print(f"Training loss: {loss}")

        test_returned = run(net, test_loader, 0, optimizer, tracker, train=False, prefix='Evaluation', epoch=e)

        results = {
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'eval': {
                'accuracy': test_returned["accuracy"],
                "precision": test_returned["precision"],
                "recall": test_returned["recall"],
                "f1-val": test_returned["F1"],
                "f1-test": test_returned["F1"]

            },
            'vocab': train_dataset.vocab,
        }
    
        torch.save(results, os.path.join(config.model_checkpoint, f"model_last.pth"))
        if test_returned["F1"] > max_f1:
            max_f1 = test_returned["F1"]
            torch.save(results, os.path.join(config.model_checkpoint, f"model_best.pth"))

        from_fold = 0

        print("+"*13)

    from_epoch = 0

    print(f"Training finished. Best F1 score on test set: {max_f1}.")
    print("="*31)

if __name__ == '__main__':
    main()
