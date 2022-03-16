# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-07 18:45:06
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
import random
import numpy as np
# import albumentations as A
from common import ImageMode


class PredictDataset(Dataset):
    def __init__(self, file_path, image_dir, resize=True, resize_height=512, resize_width=512,
                 image_mode=ImageMode.RGB, label_mode=None):
        '''
        * param file_path: 數據文件TXT：格式：image_name.jpg label_name.jpg
        * param image_dir: 圖片路徑：image_dir + image_name.jpg 構成圖片的完整路徑
        * param resize_height: 為None時，不進行缩放
        * param resize_width:  為None時，不進行缩放，
            PS：當參數resize_height或resize_width其中一個為None時，可實現等比例缩放
        '''
        self.image_label_list = self.read_file(file_path)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.resize = resize
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.image_mode = image_mode
        self.label_mode = label_mode

        self.toTensor = transforms.ToTensor()

    def __getitem__(self, i):
        index = i % self.len
        image_name, _ = self.image_label_list[index]

        image_path = os.path.join(self.image_dir, image_name)
        image = self.load_data(
            image_path, self.resize_height, self.resize_width, isimage=True)

        image = self.data_preproccess(image)
        return image, image_name

    def __len__(self):
        data_len = len(self.image_label_list)
        return data_len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回車、製表符、空格)
                content = line.rstrip().split('-')
                image_name, label_name = content
                image_label_list.append((image_name, label_name))
        return image_label_list

    def load_data(self, path, resize_height, resize_width, isimage):
        '''
            加載數據
            :param path:
            :param resize_height:
            :param resize_width:
            :return:
        '''
        if isimage:
            image = PIL.Image.open(path)
            # image is L
            if self.image_mode == ImageMode.GRAY:
                image = image.convert('L')
                assert image.mode == 'L'
            # image is RGB
            elif self.image_mode == ImageMode.RGB:
                image = image.convert('RGB')
                assert image.mode == 'RGB'
            data = image
        else:
            label = PIL.Image.open(path)
            # label is L
            if self.label_mode == ImageMode.GRAY:
                label = label.convert('L')
                assert label.mode == 'L'
            # label is RGB
            elif self.label_mode == ImageMode.RGB:
                label = label.convert('RGB')
                assert label.mode == 'RGB'
            data = label

        if self.resize:
            data = data.resize(
                (resize_height, resize_width), PIL.Image.NEAREST)
        return data

    def data_preproccess(self, image):
        image = self.toTensor(image)
        return image


class TrainDataset(PredictDataset):
    def __init__(self, file_path, image_dir, label_dir,
                 resize=True, resize_height=512, resize_width=512,
                 repeat=1, n_classes=1, trans=True, image_mode=ImageMode.RGB, label_mode=ImageMode.GRAY):
        '''
        * param filename: 數據文件TXT：格式：image_name.jpg label_name.jpg
        * param image_dir: 圖片路徑：image_dir + image_name.jpg 構成圖片的完整路徑
        * param label_dir: 圖片路徑：label_dir + label_name.jpg 構成圖片的完整路徑
        * param resize_height: 為None時，不進行缩放
        * param resize_width:  為None時，不進行缩放，
            PS：當參數resize_height或resize_width其中一個為None時，可實現等比例缩放
        * param repeat: 所有樣本數據重複次数，默認循環一次，當repeat為None時，表示無限循環<sys.maxsize
        '''
        PredictDataset.__init__(self, file_path, image_dir, resize,
                                resize_height, resize_width, image_mode=image_mode, label_mode=label_mode)
        self.label_dir = label_dir
        self.repeat = repeat
        self.n_classes = n_classes
        self.trans = trans
        """ 
            add transforms for both image and label
        """
        self.toTrans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=(-10, 10), interpolation=transforms.InterpolationMode.NEAREST),
        ])
        """ 
            add transforms for only image
        """
        self.moreTransForImage = transforms.Compose([
            transforms.RandomAdjustSharpness(sharpness_factor=2),
        ])

    def __getitem__(self, i):
        index = i % self.len
        image_name, label_name = self.image_label_list[index]

        image_path = os.path.join(self.image_dir, image_name)
        image = self.load_data(
            image_path, self.resize_height, self.resize_width, isimage=True)

        label_path = os.path.join(self.label_dir, label_name)
        label = self.load_data(
            label_path, self.resize_height, self.resize_width, isimage=False)

        image, label = self.data_preproccess(image, label)
        return image, label, image_name, label_name

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def data_preproccess(self, image, label):
        '''
            數據預處理
            :param data:
            :return:
        '''
        seed = np.random.randint(2147483647)

        def resetseed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        def toOneHot(label):
            label = label*255.0
            label = label.type(torch.LongTensor)
            label = F.one_hot(label, num_classes=self.n_classes)
            label = torch.squeeze(label, dim=0)
            label = label.permute(2, 0, 1)
            return label

        if self.trans:
            """ use same seed to let image and label do same transform """
            resetseed(seed)
            """ from numpy or PIL.Image [h, w, c] to torch.Tensor [c, h, w], and other transform """
            image = self.toTrans(image)
            image = self.moreTransForImage(image)
        else:
            image = self.toTensor(image)

        if self.trans:
            """ use same seed to let image and label do same transform """
            resetseed(seed)
            """ from numpy or PIL.Image [h, w, c] to torch.Tensor [c, h, w], and other transform """
            label = self.toTrans(label)
        else:
            label = self.toTensor(label)
        label = toOneHot(label)

        return image, label
