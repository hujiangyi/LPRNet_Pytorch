import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SingleImageDataset(Dataset):
    def __init__(self, image_path, imgSize, PreprocFun=None):
        self.image_path = image_path
        self.img_size = imgSize
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform
        if not os.path.exists(image_path):
            raise FileNotFoundError("图片文件未找到：{}".format(image_path))

    def __len__(self):
        return 1  # 数据集只包含一张图片

    def __getitem__(self, idx):
        Image = cv2.imread(self.image_path)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)
        return Image

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    # 定义转换（例如，缩放和归一化）

