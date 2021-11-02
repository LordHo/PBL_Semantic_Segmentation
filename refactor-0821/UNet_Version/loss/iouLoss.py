import torch
import numpy as np




class IOU(torch.nn.Module):
    def __init__(self, classes_average=True, weighted=False):
        super(IOU, self).__init__()
        self.classes_average = classes_average
        self.weighted = weighted

        self.default_colors = {
            'nothing'           : 1.0, # gray, augmentation
            'img_whiteside'     : 1.0, # white
            'background'        : 1.0, # black
            'bacmixgums'        : 1.0, # brown
            'artifical_crown'   : 1.0, # fluorescent blue
            'tooth'             : 1.0, # yellow
            'overlap'           : 1.0, # pink
            'cavity'            : 1.0, # red
            'cej'               : 1.0, # green
            'gums'              : 1.0, # blue
            'img_depressed'     : 1.0, # orange
            }
        self.weighted_colors = {
                  'nothing': 0.0, # gray: 0.5, setting to 0, 0.5 -> 0
            'img_whiteside': 1.0, # white: 1.0
               'background': 1.0, # black: 1.0
               'bacmixgums': 1.0, # brown: 1.5, this need to be lower, 1.5 -> 1.0
          'artifical_crown': 1.0, # fluorescent blue: 1.5, this need to be lower, 1.5 -> 1.0
                    'tooth': 2.0, # yellow: 2
                  'overlap': 1.0, # pink: 4
                   'cavity': 2.0, # red: 2
                      'cej': 2.0, # green: 2
                     'gums': 2.0, # blue: 2
            'img_depressed': 2.0, # orange: 1.5, this need to be lower, 3 -> 1.5
        }

    def forward(self, pred, target):
        if self.classes_average:
            if self.weighted:
                return self.__weighted_classes_iou(pred, target)
            else:
                return self.__classes_iou(pred, target)
        else:
            return self.__iou(pred, target)
    
    def __iou(self, pred, target):
        bs = pred.shape[0] # bs is batch size
        IoU = 0.0
        for i in range(0, bs):
            #compute the IoU of the foreground
            Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
            Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
            IoU1 = Iand1/Ior1
            #IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        return IoU/bs

    def __classes_iou(self, pred, target):
        bs, classes = pred.shape[0], pred.shape[1] # bs is batch size, classes is predict classes
        IoU = 0.0
        for i in range(0, bs):
            classes_IoU = 0.0
            for c in range(0, classes):
                Iand1 = torch.sum(target[i, c, :, :] * pred[i, c, :, :])
                Ior1 = torch.sum(target[i, c, :, :]) + torch.sum(pred[i, c, :, :]) - Iand1
                IoU1 = Iand1/Ior1
                classes_IoU = classes_IoU + (1-IoU1)
            IoU = IoU + classes_IoU/classes
        return IoU/bs

    def __weighted_classes_iou(self, pred, target):
        default_weights = np.array(list(self.default_colors.values()))
        default_weights_sum = sum(default_weights)
        weights = np.array(list(self.weighted_colors.values()))
        # TODO: check this weights can produce the old great result
        # weights             = np.array([0, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2])
        weights_sum = sum(weights)

        bs, classes = pred.shape[0], pred.shape[1] # bs is batch size, classes is predict classes
        IoU = 0.0
        for i in range(0, bs):
            classes_IoU = 0.0
            for c in range(0, classes):
                Iand1 = torch.sum(target[i, c, :, :] * pred[i, c, :, :])
                Ior1 = torch.sum(target[i, c, :, :]) + torch.sum(pred[i, c, :, :]) - Iand1
                IoU1 = Iand1/Ior1
                weight = weights[c]*(default_weights_sum/weights_sum)
                classes_IoU = classes_IoU + weight*( 1 - IoU1 )
            IoU = IoU + classes_IoU/classes
        return IoU/bs
    def print_weights(self):
        return np.array(list(self.weighted_colors.values()))

def print_WCIoU_weights():
    iou_loss = IOU(classes_average=True, weighted=True)
    return iou_loss.print_weights()

def IOU_loss(pred, label):
    iou_loss = IOU(classes_average=False)
    iou_out = iou_loss(pred, label)
    # print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out

def classes_IOU_loss(pred, label):
    iou_loss = IOU(classes_average=True)
    iou_out = iou_loss(pred, label)
    # print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out

def weighted_classes_IOU_loss(pred, label):
    iou_loss = IOU(classes_average=True, weighted=True)
    iou_out = iou_loss(pred, label)
    # print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out
