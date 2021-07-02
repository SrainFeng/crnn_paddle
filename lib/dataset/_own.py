# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：crnn_paddle 
@File    ：_own.py
@Author  ：Srain
@Date    ：2021/5/20 21:49 
"""
from __future__ import print_function, absolute_import
from paddle.io import Dataset
import os
import numpy as np
import cv2

class _OWN(Dataset):
    def __init__(self, config, is_train=True):

        super(_OWN, self).__init__()
        self.root = config.DATASET.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        with open(txt_file, 'r', encoding='utf-8') as file:
            self.labels = [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]

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

        return img, idx


import yaml
from easydict import EasyDict as edict
import lib.config.alphabets as alphabets
from paddle.io import DataLoader
import lib.utils.utils as utils

if __name__ == '__main__':

    with open("../config/OWN_config.yaml", 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    dataset = _OWN(config, is_train=False)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
    )
    for i, (inp, idx) in enumerate(train_loader()):
        # print(inp, idx)
        labels = utils.get_batch_label(dataset, idx)
        print(labels)
        if "，张海迪做到了。实" in labels:
            print('ok')
            break
