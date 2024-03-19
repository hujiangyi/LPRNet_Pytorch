# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''
import argparse
import logging

import cv2
# import torch.backends.cudnn as cudnn
import numpy as np
import torch

from data.load_data import CHARS, CHARS_DICT
from data.predict_load_data import SingleImageDataset
from model.LPRNet import build_lprnet
from torchvision import transforms
from torch.utils.data import *


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--imagefile', default="./data/test/京PL3N67.jpg", help='the predict image file name')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./weights/LPRNet__iteration_120.pth', help='pretrained base model')

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
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False
    
    if args.cuda:
        lprnet.cuda(0)
    else:
        lprnet.cpu()

    predict(lprnet, args)



def predict(Net, args):
    # 创建数据集实例
    dataset = SingleImageDataset(args.imagefile,args.img_size)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    images = next(iter(data_loader))

    if args.cuda:
        images = images.cuda()

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
        logging.info("index:" + repr(i) + ";label:" + repr(label))

if __name__ == "__main__":
    test()
