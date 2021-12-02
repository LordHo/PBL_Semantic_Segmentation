import os
from enum import Enum

import torch
import torch.optim as optim
import segmentation_models_pytorch as smp


class Mode(Enum):
    TRAIN = 'train'
    PREDICT = 'predict'


class Dataset(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class DataType(Enum):
    IMAGE = 'image'
    LABEL = 'label'


class ImageMode(Enum):
    GRAY = 'gray'
    RGB = 'RGB'


class ModelSetting:
    def __init__(self, n_classes, model_name, backbone, activation, image_mode):
        self.n_classes = n_classes
        self.model_name = model_name
        self.backbone = backbone
        self.activation = activation
        self.image_mode = image_mode
        self.in_channels = self.set_in_channels()
        self.model = self.set_model()
        self.best_model_path = None

    def set_in_channels(self):
        if self.image_mode == ImageMode.RGB:
            in_channels = 3
        elif self.image_mode == ImageMode.GRAY:
            in_channels = 1
        else:
            raise ValueError('image_mode setting value error')
        return in_channels

    def set_model(self):
        if self.model_name == 'UNet':
            model = smp.Unet(self.backbone,
                             in_channels=self.in_channels,
                             classes=self.n_classes,
                             activation=self.activation)

        elif self.model_name == 'UNet++':
            model = smp.UnetPlusPlus(self.backbone,
                                     in_channels=self.in_channels,
                                     classes=self.n_classes,
                                     activation=self.activation)

        elif self.model_name == 'PSPNet':
            model = smp.PSPNet(
                encoder_name=self.backbone,
                encoder_weights='imagenet',
                in_channels=self.in_channels,
                classes=self.n_classes,
                activation=self.activation)

        elif self.model_name == 'DeepLabv3+':
            model = smp.DeepLabV3Plus(
                encoder_name=self.backbone,
                encoder_weights='imagenet',
                in_channels=self.in_channels,
                classes=self.n_classes,
                activation=self.activation
            )

        else:
            raise NameError
        return model


class Model(ModelSetting):
    def __init__(self, n_classes, model_name, backbone, activation, image_mode):
        ModelSetting.__init__(self, n_classes, model_name,
                              backbone, activation, image_mode)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()
        self.model.to(self.device)

    def set_optimizer(self):
        # lr = 5e-4
        lr = 1e-4
        weight_decay = lr*0.1

        # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=lr, weight_decay=weight_decay)
        # JingPing
        # optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

    def set_scheduler(self):
        # max_epochs = 20
        # lambda_lr = lambda epoch: 0.8**epoch if epoch <= max_epochs else  0.8**max_epochs
        # max_epochs = 10
        # lambda_lr = lambda epoch: 0.8**(epoch//4) if (epoch//4) <= max_epochs else  0.8**max_epochs
        # lambda_lr = lambda epoch: 1.0**epoch if epoch <= max_epochs else  1.0**max_epochs
        # JingPing
        def lambda_lr(epoch):
            first_lr = 1.0
            second_lr = 0.5
            third_lr = 0.1

            if epoch < 20:
                return first_lr

            elif epoch < 50:
                return second_lr

            else:
                return third_lr

        # lambda_lr = lambda epoch: 1.0 if epoch < 20 else 0.8**((epoch-20)//3)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda_lr)
        return scheduler


def createDir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    # else:
    #     print(f'Dir: {dir_path} is already existed!')
