# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：crnn_paddle 
@File    ：train.py
@Author  ：Srain
@Date    ：2021/5/22 11:10 
"""

import argparse
from easydict import EasyDict as edict
import yaml
import os
import paddle
# import torch.backends.cudnn as cudnn
from paddle.io import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets
from lib.utils.utils import model_info

# from tensorboardX import SummaryWriter

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config

def main():

    # load config
    config = parse_arg()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # # cudnn
    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.deterministic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED

    # # writer dict
    # writer_dict = {
    #     'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
    #     'train_global_steps': 0,
    #     'valid_global_steps': 0,
    # }

    # construct face related neural networks
    model = crnn.get_crnn(config)

    # # get device
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:{}".format(config.GPUID))
    # else:
    #     device = torch.device("cpu:0")
    #
    # model = model.to(device)

    # define loss function
    criterion = paddle.nn.loss.CTCLoss()

    last_epoch = config.TRAIN.BEGIN_EPOCH

    # define optimizer and learn rate scheduler
    lr_scheduler = None
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = paddle.optimizer.lr.MultiStepDecay(
            learning_rate=config.TRAIN.LR,
            milestones=config.TRAIN.LR_STEP,
            gamma=config.TRAIN.LR_FACTOR,
            last_epoch= last_epoch - 1
        )
    else:
        lr_scheduler = paddle.optimizer.lr.StepDecay(
            learning_rate=config.TRAIN.LR,
            step_size=config.TRAIN.LR_STEP,
            gamma=config.TRAIN.LR_FACTOR,
            last_epoch=last_epoch - 1
        )
    optimizer = utils.get_optimizer(config, model, scheduler=lr_scheduler)

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = paddle.load(model_state_file)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        print("load checkpoint {}".format(model_state_file))
        model.cnn.set_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.stop_gradient = True
    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = paddle.load(model_state_file)
        if 'state_dict' in checkpoint.keys():
            model.set_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
        else:
            model.set_state_dict(checkpoint)

    model_info(model)

    train_dataset = get_dataset(config)(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS
    )

    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS
    )

    best_acc = 0
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, epoch)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)

        # save checkpoint
        paddle.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc
            }, os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pdparams".format(epoch, acc))
        )


if __name__ == '__main__':
    main()

