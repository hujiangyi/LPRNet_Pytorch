from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
    # , '学', '港', '使', '领', '澳', '挂', '临'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, base_path, txt_file, imgSize, lpr_max_len, PreprocFun=None):
        self.img_paths = []
        self.labels = []
        with open(base_path + txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if self.skip_line(parts):
                    print("skip label:", parts[1])
                    continue
                self.img_paths.append(base_path + parts[0])
                self.labels.append(parts[1])

        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        label_str = self.labels[index]

        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        label = []
        for c in label_str:
            label.append(CHARS_DICT[c])

        # if len(label) == 8 or len(label) == 9 or len(label) == 10:
        #     if not self.check(label):
        #         print(filename,label_str)
        #         assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True

    def skip_line(self, parts):
        skip_characters = {'学', '港', '使', '领', '澳', '挂', '临'}
        for char in parts[1]:
            if char in skip_characters:
                return True
        return False
