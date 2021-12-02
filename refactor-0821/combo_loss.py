import torch
import torch.nn as nn


# PyTorch
BETA = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
ALPHA = 0.5  # weighted contribution of modified CE loss compared to Dice loss


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        # dice loss
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        # modified cross entropy
        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (BETA * ((targets * torch.log(inputs)) +
                 ((1 - BETA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        # print(out)
        weighted_ce = out.mean(-1)

        # print(dice, weighted_ce)
        combo = (ALPHA * weighted_ce) - ((1 - ALPHA) * dice)

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
