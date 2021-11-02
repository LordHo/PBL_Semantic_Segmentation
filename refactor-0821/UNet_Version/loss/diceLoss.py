from segmentation_models_pytorch.losses import DiceLoss
import numpy as np

"""
    predict.shape is (4, 11, 512, 512), dtype is tensor or numpy
    truth.shape is (4, 11, 512, 512), dtype is tensor or numpy
"""

def separate_dice_loss(predict, truth, class_weights=None):
    diceLoss = DiceLoss(mode='binary')
    
    (bs, c, h, w) = predict.shape

    default_weights_sum = sum(np.ones(c))
    if not(class_weights is None):
        class_weights_sum = sum(class_weights)

    total_dice_loss = 0.0
    for i_b in range(bs):
        for i_c in range(c):
            if not(class_weights is None): 
                total_dice_loss += diceLoss(predict[i_b, i_c, :, :], truth[i_b, i_c, :, :]) * class_weights[i_c]
            else:
                total_dice_loss += diceLoss(predict[i_b, i_c, :, :], truth[i_b, i_c, :, :])
    
    return (total_dice_loss/(bs*c)) if class_weights is None else (total_dice_loss/(bs*c))*(default_weights_sum/class_weights_sum)