# coding=utf-8
# !/usr/bin/env python
"""
@Project ：crnn_paddle
@File    ：_360cc.py
@Author  ：Srain
@Date    ：2021/5/20 16:58
"""

from __future__ import print_function, absolute_import
from paddle.io import Dataset
import paddle
import os
import numpy as np
import cv2


class _360CC(Dataset):
    def __init__(self, config, is_train=True):

        super(_360CC, self).__init__()
        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        char_file = config.DATASET.CHAR_FILE
        with open(char_file, 'rb') as file:
            char_dict = {num: char.strip().decode('utf-8', 'ignore') for num, char in enumerate(file.readlines())}

        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        self.labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                imgname = c.split(' ')[0]
                indices = c.split(' ')[1:]
                string = ''.join([char_dict[int(idx)] for idx in indices])
                self.labels.append({imgname: string})

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        img = cv2.imread(os.path.join(self.root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_h, img_w = img.shape

        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        # img = paddle.to_tensor(img)
        return img, idx

    def get_img_name(self, idx):
        return list(self.labels[idx].keys())[0]


import yaml
from easydict import EasyDict as edict
import lib.config.alphabets as alphabets
from paddle.io import DataLoader
import lib.utils.utils as utils
if __name__ == '__main__':

    with open("../config/360CC_config.yaml", 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    config.DATASET.CHAR_FILE = '../dataset/txt/360cc/char_std_5990_utf.txt'
    config.DATASET.JSON_FILE['train'] = '../dataset/txt/360cc/train.txt'

    dataset = _360CC(config, is_train=True)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
    )
    total = len(train_loader())
    lab_ans = []
    ans = []
    for i, (inp, idx) in enumerate(train_loader()):
        # print(inp, idx)
        labels = utils.get_batch_label(dataset, idx)
        print(i, "/", total, labels)

        for j in range(len(labels)):
            if '\ue004' in labels[j]:
                lab_ans.append([labels[j]])
                ans.append(idx[j])
        # if "，张海迪做到了。实" in labels:
        #     print('ok', idx)
        #     break
    print(lab_ans)
    print(ans)
    for an in ans:
        print(dataset.get_img_name(an))
    # img_name = dataset.get_img_name(924778)
    # print(img_name)
    # print(utils.get_batch_label(dataset, [924778]))
    # print(list(dataset.labels[924778].values())[0])
    # print(utils.get_batch_label(dataset, [1, 395, 177, 1240, 244, 34, 14, 3, 5535, 101]))