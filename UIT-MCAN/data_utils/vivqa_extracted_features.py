import torch
from torch.utils import data
from torch.utils.data.dataset import random_split
from .utils import preprocess_question, preprocess_answer
from .vocab import Vocab
import h5py
import json
import numpy as np
from PIL import Image
from glob import glob
from torchvision import transforms as tf
import config

class ViVQA(data.Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, json_path, vocab=None, image_dir=None):
        super(ViVQA, self).__init__()
        with open(json_path, 'r') as fd:
            json_data = json.load(fd)

        # vocab
        self.vocab = Vocab([json_path]) if vocab is None else vocab
        # q and a
        self.questions, self.answers, self.image_ids = self.load_json(json_data)

        # v
        self.image_dir = image_dir
        # self.image_id_to_index = self._create_image_id_to_index()

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions)) + 2
        return self._max_length + 2

    @property
    def num_tokens(self):
        return len(self.vocab.stoi)

    def load_json(self, json_data):
        questions = []
        answers = []
        image_ids = []
        for ann in json_data["annotations"]:
            questions.append(preprocess_question(ann["question"]))
            answers.append(preprocess_answer(ann["answer"]))
            image_ids.append(ann["img_id"])

        return questions, answers, image_ids

    # def _create_image_id_to_index(self):
    #     """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
    #     with h5py.File(self.image_features_path, 'r') as features_file:
    #         coco_ids = features_file['ids'][()]
    #     coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
    #     return coco_id_to_index

    def _load_image(self, image_id):
        """ Load an image """
        # if not hasattr(self, 'features_file'):
        #     # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
        #     # forks for multiple works, every child would use the same file object and fail
        #     # Having multiple readers using different file objects is fine though, so we just init in here.
        #     self.features_file = h5py.File(self.image_features_path, 'r')
        # index = self.image_id_to_index[image_id]
        # dataset = self.features_file['features']
        # img = dataset[index].astype('float32')
        image_path = glob(self.image_dir + f'/*0{image_id}.jpg')[0]
        image = Image.open(image_path)


        image = tf.ToTensor()(tf.Resize(config.image_size)(image))
        return image

    def __getitem__(self, idx):
        q = self.vocab._encode_question(self.questions[idx])
        a = self.vocab._encode_answer(self.answers[idx])
        image_id = self.image_ids[idx]
        v = self._load_image(image_id)
        if v.shape[0] == 1:
          v = torch.concat([v]*3, dim=0)
        return v, q, a

    def __len__(self):
        return len(self.questions)


def get_loader(train_dataset, test_dataset=None, k_fold=True):
    """ Returns a data loader for the desired split """

    if k_fold:
        fold_size = int(len(train_dataset) * 0.2)

        subdatasets = random_split(train_dataset, [fold_size, fold_size, fold_size, fold_size, len(train_dataset) - fold_size*4], generator=torch.Generator().manual_seed(13))
        
        folds = []
        for subdataset in subdatasets:
            folds.append(
                torch.utils.data.DataLoader(
                    subdataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=config.data_workers))

        if test_dataset:
            test_fold = torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=config.data_workers)

            return folds, test_fold

        return folds
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=config.data_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=32,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.data_workers)
                
        return train_loader, test_loader
if __name__ == '__main__':
    json_path = "/home/ubuntu/learning/VQA/UIT-MCAN/ViVQA/vivqa_train_2017.json"
    image_dir = "/home/ubuntu/learning/VQA/UIT-MCAN/ViVQA/images/train"
    train_dataset = ViVQA(json_path, None, image_dir)
    for i in range(len(train_dataset)):
        train_dataset[i]