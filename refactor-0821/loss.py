import numpy as np

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from UNet_Version.loss.bceLoss import BCE_loss
from UNet_Version.loss.iouLoss import IOU_loss, classes_IOU_loss, weighted_classes_IOU_loss, print_WCIoU_weights
from UNet_Version.loss.focalLoss import separate_focal_loss
from UNet_Version.loss.diceLoss import separate_dice_loss


class Loss:
    def __init__(self):
        self.resetall()
        self.setting()

    def resetall(self):
        self.seg_loss = 0.0
        self.bce_loss = 0.0
        self.focal_loss = 0.0
        self.dice_loss = 0.0
        self.iou_loss = 0.0
        self.classes_iou_loss = 0.0
        self.weighted_classes_iou_loss = 0.0
        self.ce_loss = 0.0

    def setting(self):
        colors = {
            'nothing': 0.0,  # gray: 0.5, setting to 0, 0.5 -> 0
            'img_whiteside': 1.0,  # white: 1.0
            'background': 1.0,  # black: 1.0
            'bacmixgums': 1.0,  # brown: 1.5, this need to be lower, 1.5 -> 1.0
            'artifical_crown': 1.0,  # fluorescent blue: 1.5, this need to be lower, 1.5 -> 1.0
            'tooth': 2.5,  # yellow: 2
            'overlap': 1.0,  # pink: 4
            'cavity': 2.5,  # red: 2
            'cej': 2.5,  # green: 2
            'gums': 2.5,  # blue: 2
            'img_depressed': 2.5,  # orange: 1.5, this need to be lower, 3 -> 1.5
        }
        # self.dice_weights = np.array(list(colors.values()))
        # JingPing
        self.dice_weights = np.array([0.5, 1, 1, 1.5, 1.5, 3, 2, 2, 3, 2, 2])
        # TODO: check this weights can produce the old great result
        # self.dice_weights = np.array([0.5, 1, 1, 1.5, 1.5, 2, 4, 2, 2, 2, 3])
        # self.dice_weights = np.ones(11)
        # self.dice_weights[0] = 0
        self.useBCE = False
        self.useFocal = True
        self.useDice = True
        self.useIoU = True
        self.useClassesIoU = False
        self.useWeightedClassesIoU = False

        self.focal_loss_function = smp.losses.FocalLoss(mode='multiclass')
        self.dice_loss_function = smp.losses.DiceLoss(
            mode='multiclass', classes=self.dice_weights)

        self.useCE = True
        self.cross_entropy = nn.CrossEntropyLoss()

    def sum_loss(self):
        if self.useBCE:
            self.seg_loss += self.bce_loss
        if self.useFocal:
            self.seg_loss += self.focal_loss
        if self.useDice:
            self.seg_loss += self.dice_loss
        if self.useIoU:
            # self.seg_loss += self.iou_loss
            self.seg_loss += 0.0
        if self.useClassesIoU:
            self.seg_loss += self.classes_iou_loss
        if self.useWeightedClassesIoU:
            self.seg_loss += self.weighted_classes_iou_loss
        if self.useCE:
            self.seg_loss += self.ce_loss

    def trans_y(self, y):
        y_ = y.type(torch.cuda.LongTensor)
        y_ = y_.argmax(dim=1, keepdim=True)
        y_ = np.squeeze(y_, axis=1)
        return y_

    def calculate_total_loss(self, output, y):
        self.resetall()
        y_ = self.trans_y(y)
        if self.useBCE:
            self.calculate_bce_loss(output, y)
        if self.useFocal:
            self.calculate_focal_loss(output, y_)
            # self.calculate_separate_focal_loss(output, y)
        if self.useDice:
            self.calculate_dice_loss(output, y_)
            # self.calculate_separate_dice_loss(output, y)
        if self.useIoU:
            self.calculate_iou_loss(output, y)
        if self.useClassesIoU:
            self.calculate_classes_iou_loss(output, y)
        if self.useWeightedClassesIoU:
            self.calculate_weighted_classes_iou_loss(output, y)
        if self.useCE:
            self.calculate_ce_loss(output, y)

        self.sum_loss()

        return self.seg_loss

    def calculate_bce_loss(self, output, y):
        self.bce_loss = BCE_loss(output, y)

    def calculate_focal_loss(self, output, y_):
        self.focal_loss = self.focal_loss_function(output, y_)

    def calculate_separate_focal_loss(self, output, y):
        self.focal_loss = separate_focal_loss(output, y)

    def calculate_dice_loss(self, output, y_):
        self.dice_loss = self.dice_loss_function(output, y_)

    def calculate_separate_dice_loss(self, output, y):
        self.dice_loss = separate_dice_loss(
            output, y, class_weights=self.dice_weights)

    def calculate_iou_loss(self, output, y):
        self.iou_loss = IOU_loss(output, y)
        # due to no use IoU but the grad will still occupy GPU memory and cause run out of memory
        self.iou_loss = self.iou_loss.detach()

    def calculate_classes_iou_loss(self, output, y):
        self.classes_iou_loss = classes_IOU_loss(output, y)

    def calculate_weighted_classes_iou_loss(self, output, y):
        self.weighted_classes_iou_loss = weighted_classes_IOU_loss(output, y)

    def calculate_ce_loss(self, output, y):
        self.ce_loss = self.cross_entropy(output, y)

    def print_used_losses(self):
        losses = []
        if self.useBCE:
            losses.append('BCE')
        if self.useFocal:
            losses.append('Focal')
        if self.useDice:
            losses.append('Dice')
        if self.useIoU:
            losses.append('IoU')
        if self.useClassesIoU:
            losses.append('CIoU')
        if self.useWeightedClassesIoU:
            losses.append('WCIoU')
        if self.useCE:
            losses.append('CE')
        message = ', '.join(losses)
        return message

    def print_dice_weights(self):
        return ', '.join(self.dice_weights.astype(str))

    def print_WCIoU_weights(self):
        return ', '.join(print_WCIoU_weights().astype(str))
