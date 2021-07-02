# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：crnn_paddle
@File    ：function.py
@Author  ：Srain
@Date    ：2021/5/22 11:10
"""
from __future__ import absolute_import
import time
import lib.utils.utils as utils
import paddle
import sys
import os

class AverageMeter(object):
    """Compute and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, dataset, converter, model, criterion, optimizer, epoch, writer_dict=None, output_dict=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader()):
        data_time.update(time.time() - end)

        # inp.place = paddle.CUDAPlace(0)
        # print(inp)
        labels = utils.get_batch_label(dataset, idx)
        inp.cuda()
        # inp = inp.to(device)

        # inference
        preds = model(inp).cpu()

        # compute loss
        batch_size = inp.shape[0]
        text, length = converter.encode(labels)
        preds_size = paddle.to_tensor([preds.shape[0]] * batch_size, dtype="int64")
        # print(text.shape)
        # print(idx, labels)
        text = paddle.reshape(text, [batch_size, -1])
        loss = criterion(preds, text, preds_size, length)
        # print(loss.numpy()[0])

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.numpy()[0], inp.shape[0])

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader()), batch_time=batch_time,
                      speed=inp.shape[0]/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

            # if writer_dict:
            #     writer = writer_dict['writer']
            #     global_steps = writer_dict['train_global_steps']
            #     writer.add_scalar('train_loss', losses.avg, global_steps)
            #     writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


@paddle.no_grad()
def validate(config, val_loader, dataset, converter, model, criterion, epoch, writer_dict=None, output_dict=None):
    print(sys.stdout.encoding)

    global preds_size, preds, sim_preds, labels
    losses = AverageMeter()
    model.eval()

    n_correct = 0
    for i, (inp, idx) in enumerate(val_loader()):

        labels = utils.get_batch_label(dataset, idx)

        # inference
        preds = model(inp).cpu()

        # compute loss
        batch_size = inp.shape[0]
        text, length = converter.encode(labels)
        preds_size = paddle.to_tensor([preds.shape[0]] * batch_size, dtype="int64")
        loss = criterion(preds, text, preds_size, length)

        losses.update(loss.numpy()[0], inp.shape[0])

        preds = preds.argmax(axis=2)

        preds = paddle.reshape(preds.transpose([1, 0]), [-1])

        sim_preds = converter.decode(preds, preds_size, raw=False)
        for pred, target in zip(sim_preds, labels):
            if pred == target:
                n_correct += 1

        if (i + 1) % config.PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

        if i == config.TEST.NUM_TEST_BATCH:
            break

    raw_preds = converter.decode(preds, preds_size, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred.encode('utf-8'), pred.encode('utf-8'), gt.encode('utf-8')))
        # print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        # print("{:<20} => {:<20}, gt: {:<20}".format(raw_pred, pred, gt))

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)

    print("[#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    accuracy = n_correct / float(num_test_sample)
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    # if writer_dict:
    #     writer = writer_dict['writer']
    #     global_steps = writer_dict['valid_global_steps']
    #     writer.add_scalar('valid_acc', accuracy, global_steps)
    #     writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy


@paddle.no_grad()
def eval(val_loader, dataset, converter, model):
    model.eval()
    try:
        output_f = open(os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../..')), "eval_result.txt"), 'x', encoding='utf-8')
        for i, (inp, idx) in enumerate(val_loader()):
            preds = model(inp).cpu()

            batch_size = inp.shape[0]
            preds_size = paddle.to_tensor([preds.shape[0]] * batch_size, dtype="int64")

            preds = preds.argmax(axis=2)
            preds = paddle.reshape(preds.transpose([1, 0]), shape=[-1])
            print('preds: ', preds)
            print('preds_size: ', preds_size)
            sim_preds = converter.decode(preds, preds_size, raw=False)
            print('sim_preds: ', sim_preds)

            for j in range(len(sim_preds)):
                # print(idx[j])
                file_name = dataset.get_img_name(idx[j])
                print(file_name, sim_preds[j])
                output_f.write("{}\t{}\n".format(file_name, sim_preds[j]))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    import sys
    print(sys.stdout.encoding)
