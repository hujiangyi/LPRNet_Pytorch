# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os
import logging

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=15, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--base_data_dirs', default="../CBLPRD-330k_v1/", help='the test data base path')
    parser.add_argument('--train_txt', default="train.txt", help='the train images config file')
    parser.add_argument('--test_txt', default="train.txt", help='the test images config file')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.1, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=100, help='training batch size.')
    parser.add_argument('--test_batch_size', default=1000, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=500, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=6000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(int)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def train():
    logging.basicConfig(filename='out.txt', level=logging.INFO)
    args = get_parser()

    T_length = 18 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    # lprnet.to(device)
    logging.info("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        logging.info("load pretrained model successful!")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        logging.info("initial net weights successful!")

    if bool(args.cuda):
        lprnet.cuda(0)
    else:
        lprnet.cpu()

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    train_dataset = LPRDataLoader(args.base_data_dirs, args.train_txt, args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(args.base_data_dirs, args.test_txt, args.img_size, args.lpr_max_len)

    epoch_size = len(train_dataset) // int(args.train_batch_size)
    max_iter = args.max_epoch * epoch_size

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, int(args.train_batch_size), shuffle=True,num_workers=int(args.num_workers), collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        if iteration !=0 and iteration % int(args.save_interval) == 0:
            logging.info('torch.save ' + repr(iteration))
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth')

        if (iteration + 1) % int(args.test_interval) == 0:
            logging.info('Greedy_Decode_Eval ' + repr(iteration))
            Greedy_Decode_Eval(lprnet, test_dataset, args)
            # lprnet.train() # should be switch to train mode

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # logging.info(labels)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if bool(args.cuda):
            images = images.cuda()
            labels = images.cuda()

        # forward
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # logging.info(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.detach().requires_grad_()
        # logging.info(log_probs.shape)
        # backprop
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        if loss.item() == np.inf:
            continue
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
        if iteration % 20 == 0:
            logging.info('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))
    # final test
    logging.info("Final test Accuracy:")
    Greedy_Decode_Eval(lprnet, test_dataset, args)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // int(args.test_batch_size)
    batch_iterator = iter(DataLoader(datasets, int(args.test_batch_size), shuffle=True, num_workers=int(args.num_workers), collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for e in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length

        max_length = max(lengths)  # 计算所有标签序列的最大长度
        # 将每个标签序列填充到相同的长度
        for i in range(len(targets)):
            current_length = len(targets[i])
            if current_length < max_length:
                padding_length = max_length - current_length
                # 在标签序列末尾填充 0
                targets[i] = np.concatenate((targets[i], [len(CHARS) - 1] * padding_length))
            elif current_length > max_length:
                # 如果标签序列长度超过最大长度，则截断
                targets[i] = targets[i][:max_length]

        # # 将填充或截断后的标签序列转换为 NumPy 数组
        targets = np.array(targets)

        # targets = np.array([el.numpy() for el in targets])

        if bool(args.cuda):
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

        if e % 20 == 0:
            logging.info('test epoch: ' + repr(e) + '|| Totel ' + repr(epoch_size))

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    logging.info("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    logging.info("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))


if __name__ == "__main__":
    train()
