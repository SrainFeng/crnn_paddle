# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：crnn_paddle 
@File    ：eval_dataset.py
@Author  ：Srain
@Date    ：2021/6/15 16:47 
"""
from __future__ import print_function, absolute_import
from paddle.io import Dataset
import os
import numpy as np
import cv2


class EvalDataset(Dataset):
    def __init__(self, path, inp_h, inp_w, gt_txt=None):
        """

        :param path: 测试数据集的目录
        :param inp_h: 输入的图片高
        :param inp_w: 输入的图片宽
        :param gt_txt: gt文件的路径
        """

        super(EvalDataset, self).__init__()
        self.root = path
        self.inp_h = inp_h
        self.inp_w = inp_w

        self.txt_file = path
        self.gt_txt = gt_txt

        self.mean = 0.588
        self.std = 0.193

        if gt_txt is not None:
            with open(gt_txt, 'r', encoding='utf-8') as file:
                self.labels = [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]

        self.images = self._get_file_paths()

        self.images.sort()

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_h, img_w = img.shape

        img = cv2.resize(img, (0, 0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img / 255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx

    def _get_file_paths(self, extension_name='jpg'):
        ans = []
        file_dir = self.root
        for root, dirs, files in os.walk(file_dir):
            # print(root, dirs, files)
            for file in files:
                if os.path.splitext(file)[1] == '.' + extension_name:
                    ans.append(os.path.join(root, file))
        return ans

    def get_img_name(self, idx):
        _, img_name = os.path.split(self.images[idx])
        return img_name
