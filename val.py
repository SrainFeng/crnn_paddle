# !/usr/bin/env python
# -*- coding: UTF-8 -*-
# encoding=utf8
"""
@Project ：crnn_paddle 
@File    ：val.py
@Author  ：Srain
@Date    ：2021/6/15 15:55 
"""

import argparse
import paddle
from paddle.io import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset.eval_dataset import EvalDataset
from lib.core import function
import lib.config.alphabets as alphabets
from lib.utils.utils import model_info
import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def parse_arg():
    parser = argparse.ArgumentParser(description="crnn val")

    parser.add_argument('--checkpoint', type=str, default='output/checkpoints/mixed_second_finetune_acc_97P7.pth',
                        help='the path to your checkpoints')
    parser.add_argument('--val_dataset', type=str, help='the path of eval images')

    args = parser.parse_args()

    return args

def main():
    config = parse_arg()

    img_h = 32
    img_w = 160
    alphabet = alphabets.alphabet
    n_class = len(alphabet)
    n_h = 256

    model = crnn.get_val_crnn(img_h, n_class, n_h)

    model_state_file = config.checkpoint
    if model_state_file == '':
        print(" => no checkpoint found")
    checkpoint = paddle.load(model_state_file)
    if 'state_dict' in checkpoint.keys():
        model.set_state_dict(checkpoint['state_dict'])
        last_epoch = checkpoint['epoch']
    else:
        model.set_state_dict(checkpoint)

    model_info(model)

    val_dataset = EvalDataset(config.val_dataset, img_h, img_w)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    converter = utils.strLabelConverter(alphabet)
    function.eval(val_loader, val_dataset, converter, model)


if __name__ == '__main__':
    main()

