import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp


def flatten(tensor):
    # From https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/unet3d
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def GeneralizedDiceLoss(input, target):
    epsilon = 1e-6
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    if input.size(0) == 1:
        # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
        # put foreground and background voxels in separate channels
        input = torch.cat((input, 1 - input), dim=0)
        target = torch.cat((target, 1 - target), dim=0)

    # GDL weighting: the contribution of each label is corrected by the inverse of its volume
    w_l = target.sum(-1)
    w_l = 1. / (w_l * w_l).clamp(min=epsilon)
    w_l.requires_grad = False

    intersect = (input * target).sum(-1)
    intersect = intersect * w_l

    denominator = (input + target).sum(-1)
    denominator = (denominator * w_l).clamp(min=epsilon)

    return 1. - torch.mean(2. * (intersect.sum() / denominator.sum()))


def GeneralizedDiceLossWithMask(input, target):
    epsilon = 1e-6
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    if input.size(0) == 1:
        # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
        # put foreground and background voxels in separate channels
        input = torch.cat((input, 1 - input), dim=0)
        target = torch.cat((target, 1 - target), dim=0)

    # GDL weighting: the contribution of each label is corrected by the inverse of its volume
    w_l = target.sum(-1)
    mask = w_l > 0
    # mask out the emtpy class in ground truth
    w_l = 1. / (w_l * w_l).clamp(min=epsilon)
    w_l.requires_grad = False

    intersect = (input * target).sum(-1)
    intersect = intersect * w_l
    intersect *= mask

    denominator = (input + target).sum(-1)
    denominator = (denominator * w_l).clamp(min=epsilon)
    denominator *= mask

    return 1. - torch.mean(2. * (intersect.sum() / denominator.sum()))


def DiceLoss(inputs, targets, smooth=1):
    intersection = (inputs * targets).sum()
    denominator = inputs.sum() + targets.sum()
    return 1. - (2. * intersection + smooth) / (denominator + smooth)


# PyTorch
BETA = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
ALPHA = 0.5  # weighted contribution of modified CE loss compared to Dice loss
ce_eps = 1e-15


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()
        self.out_last = None
        self.GDL = True  # GDL is Generalized Dice Loss, false is using dice loss
        self.MCE = False  # MCE is Modified Cross Entropy
        self.Focal = True
        self.checkNaN = False

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        # True Positives, False Positives & False Negatives
        # dice loss
        if self.GDL:
            # dice = GeneralizedDiceLoss(inputs, targets)
            dice = GeneralizedDiceLossWithMask(inputs, targets)
        else:
            dice = DiceLoss(inputs, targets, smooth)

        if self.Focal:
            def trans_y(y):
                y_ = y.type(torch.cuda.LongTensor)
                y_ = y_.argmax(dim=1, keepdim=True)
                y_ = np.squeeze(y_, axis=1)
                return y_

            focal = smp.losses.FocalLoss(mode='multiclass')
            out = focal(inputs, trans_y(targets))
        elif self.MCE:
            # modified cross entropy
            ce_function = nn.CrossEntropyLoss()
            out = (BETA * ce_function(inputs, targets) +
                   (1 - BETA) * ce_function(1.0 - inputs, 1.0 - targets))
        else:
            ce_function = nn.CrossEntropyLoss()
            out = ce_function(inputs, targets)

        ce = out.mean(-1)

        # original version, but loss become nan in some cases
        # out = - (BETA * ((targets * torch.log(inputs)) +
        #          ((1 - BETA) * (1.0 - targets) * torch.log(1.0 - inputs))))

        if self.checkNaN:
            assert torch.any(torch.isnan(
                out)) == False, f'nan number is {out[torch.isnan(out)].size()}.'
            if torch.any(torch.isnan(out)):
                print(self.out_last[torch.isnan(out)])
                print(out[torch.isnan(out)])
            self.out_last = out

        combo = (ALPHA * ce) + ((1 - ALPHA) * dice)

        return combo


def test_combo_loss():
    combo_loss = ComboLoss()
    # ones = torch.ones((1, 3))
    # zeros = torch.zeros((1, 3))
    # inputs = torch.cat((ones, ones, ones, zeros, zeros, zeros), 0)
    inputs = torch.rand(3, 3)
    print('inputs:\n'+'-'*20)
    print(inputs)
    # tragets = torch.cat((ones, ones, ones, ones, ones, ones), 0)
    tragets = torch.rand(3, 3)
    print('tragets:\n'+'-'*20)
    print(tragets)

    loss = combo_loss(inputs, tragets)
    print(f'combo loss: {loss:.4f}')


if __name__ == '__main__':
    test_combo_loss()
