from segmentation_models_pytorch.losses import FocalLoss
# import numpy as np

"""
    predict.shape is (4, 11, 512, 512), dtype is tensor or numpy
    truth.shape is (4, 11, 512, 512), dtype is tensor or numpy
"""

def separate_focal_loss(predict, truth):
    focalLoss = FocalLoss(mode='binary')
    
    (bs, c, h, w) = predict.shape

    total_focal_loss = 0.0
    for i_b in range(bs):
        for i_c in range(c):
            total_focal_loss += focalLoss(predict[i_b, i_c, :, :], truth[i_b, i_c, :, :])
    
    return (total_focal_loss/(bs*c))