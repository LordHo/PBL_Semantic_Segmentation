from tqdm import tqdm

from torch.utils.data import DataLoader

from common import *
from dataset import *
from loss import Loss
from test import testing

class EpochInfo:
    def __init__(self, setting_epochs):
        self.epochs = setting_epochs
        self.early_stop = False
        self.epochs_no_progress = 0
        self.best_iou_loss = float('inf')
        self.train_loss_history = []
        self.resetepochloss()
        self.reset()
        self.log = None

    def reset(self):
        self.save_model = False
    
    def resetepochloss(self):
        self.training_loss = 0.0
        self.epoch_bce_loss = 0.0
        self.epoch_focal_loss = 0.0
        self.epoch_dice_loss = 0.0
        self.epoch_iou_loss = 0.0
        self.epoch_classes_iou_loss = 0.0
        self.epoch_weighted_classes_iou_loss = 0.0
        self.epoch_ce_loss = 0.0

    def epoch_message(self, epoch_info, batch_num, epoch, last_lr):
        list_ = [f"Epoch {epoch:2d}"]

        print_training_loss = False
        if self.training_loss > 0.0 and print_training_loss:
            train_loss_message = f'Training loss: {epoch_info.training_loss/batch_num:.3f}'
            list_.append(train_loss_message)

        if self.epoch_bce_loss > 0.0:
            bce_loss_message   = f'Bce: {epoch_info.epoch_bce_loss/batch_num:.3f}'
            list_.append(bce_loss_message)

        if self.epoch_focal_loss > 0.0: 
            focal_message  = f'Focal: {epoch_info.epoch_focal_loss/batch_num:.3f}'
            list_.append(focal_message)

        if self.epoch_dice_loss > 0.0:
            dice_message   = f'Dice: {epoch_info.epoch_dice_loss/batch_num:.3f}'
            list_.append(dice_message)

        if self.epoch_iou_loss > 0.0:
            print_iou_loss = False
            if print_iou_loss:
                iou_loss_message   = f'IoU loss: {epoch_info.epoch_iou_loss/batch_num:.3f}'
                list_.append(iou_loss_message)
            else:
                iou_message        = f'IoU: {1 - epoch_info.epoch_iou_loss/batch_num:.3f}'
                list_.append(iou_message)

        if self.epoch_classes_iou_loss > 0.0:
            print_classes_iou_loss = False
            if print_classes_iou_loss:
                classes_iou_loss_message = f'CIoU loss: {epoch_info.epoch_classes_iou_loss/batch_num:.3f}'
                list_.append(classes_iou_loss_message)
            else:
                classes_iou_message = f'CIoU: {1-epoch_info.epoch_classes_iou_loss/batch_num:.3f}'
                list_.append(classes_iou_message)

        if self.epoch_weighted_classes_iou_loss > 0.0:
            print_weighted_classes_iou_loss = False
            if print_weighted_classes_iou_loss:
                weighted_classes_iou_loss = f'WCIoU loss: {epoch_info.epoch_weighted_classes_iou_loss/batch_num:.3f}'
                list_.append(weighted_classes_iou_loss)
            else:
                weighted_classes_iou = f'WCIoU: {1 - epoch_info.epoch_weighted_classes_iou_loss/batch_num:.3f}'
                list_.append(weighted_classes_iou)

        if self.epoch_ce_loss > 0.0:
            ce_message  = f'CE: {epoch_info.epoch_ce_loss/batch_num:.3f}'
            list_.append(ce_message)

        last_lr_message = f'lr: {last_lr:.1e}'
        list_.append(last_lr_message)
        
        message = ', '.join(list_)
        return message

def epoch_training(epoch, model, dataloader, loss, epoch_info):
    import time

    epoch_info.resetepochloss()
    epoch_info.reset()

    pbar = tqdm(dataloader)
    for batch_index, (batch_image, batch_label, _, _) in enumerate(pbar):
        # time.sleep(0.5)
        x, y = batch_image.to(model.device), batch_label.to(model.device)
        y = y.type(torch.cuda.FloatTensor)

        model.optimizer.zero_grad() # reset the gradient to zero
        output = model.model(x)
        loss.calculate_total_loss(output, y)
        """ Zero gradients, perform a backward pass, and update the weights. """
        loss.seg_loss.backward()
        model.optimizer.step()

        last_lr = model.scheduler.get_last_lr()[0]

        epoch_info.training_loss += loss.seg_loss.item()
        epoch_info.epoch_bce_loss += loss.bce_loss
        epoch_info.epoch_focal_loss += loss.focal_loss
        epoch_info.epoch_dice_loss += loss.dice_loss
        epoch_info.epoch_iou_loss += loss.iou_loss
        epoch_info.epoch_classes_iou_loss += loss.classes_iou_loss
        epoch_info.epoch_weighted_classes_iou_loss += loss.weighted_classes_iou_loss

        epoch_info.epoch_ce_loss += loss.ce_loss
        
        batch_num = batch_index+1

        message = epoch_info.epoch_message(epoch_info, batch_num, epoch, last_lr)
        
        pbar.set_description(message)
        pbar.refresh()
    print(message, file=epoch_info.log)

    if epoch_info.best_iou_loss > epoch_info.epoch_iou_loss + epoch_info.epoch_classes_iou_loss:
        epoch_info.best_iou_loss = epoch_info.epoch_iou_loss + epoch_info.epoch_classes_iou_loss
        epoch_info.epochs_no_progress = 0
        epoch_info.save_model = True
    else:
        epoch_info.epochs_no_progress += 1

    epoch_info.train_loss_history.append(epoch_info.training_loss)

    return model, epoch_info


def training(file_path, image_dir, label_dir, result_dir, model, image_mode, label_mode):
    
    print(f'Device: {model.device}')

    dataset_pars = {
        'file_path'     : file_path,
        'image_dir'     : image_dir, 
        'label_dir'     : label_dir, 
        'resize'        : True,
        'resize_height' : 512,
        'resize_width'  : 512,
        'repeat'        : 20, 
        # 'repeat': 1, # for test use
        'n_classes'     : 10+1, 
        'trans'         : True,
        'image_mode'    : image_mode,
        'label_mode'    : label_mode,
    }
    train_dateset = TrainDataset(**dataset_pars)

    train_loader_pars = {
        'dataset'   : train_dateset,
        'batch_size': 4,
        'shuffle'   : False
    }
    train_dataloader = DataLoader(**train_loader_pars)

    loss = Loss()

    # epoch_info = EpochInfo(setting_epochs=75)
    epoch_info = EpochInfo(setting_epochs=100)
    # epoch_info = EpochInfo(setting_epochs=1) # for test use
    repeat = dataset_pars['repeat']
    batch_size = train_loader_pars['batch_size']
    image_shape = (dataset_pars['resize_height'], dataset_pars['resize_width'])
    message = f'Training epochs: {epoch_info.epochs}, repeat: {repeat}, batch size: {batch_size}, image shape (height, width): {image_shape}'
    print(message)
    lof_path = os.path.join(result_dir, 'log.txt')
    epoch_info.log = open(lof_path, 'w')
    print(message, file=epoch_info.log)

    use_losses = loss.print_used_losses()
    print(f'Used losses: {use_losses}.')
    print(f'Used losses: {use_losses}.', file=epoch_info.log)
    if loss.useDice:
        dice_weights = loss.print_dice_weights()
        print(f'Dice weights: {dice_weights}.')
        print(f'Dice weights: {dice_weights}.', file=epoch_info.log)
    if loss.useWeightedClassesIoU:
        WCIoU_weights = loss.print_WCIoU_weights()
        print(f'WCIoU weights: {WCIoU_weights}.')
        print(f'WCIoU weights: {WCIoU_weights}.', file=epoch_info.log)
    
    early_stop_limit = epoch_info.epochs
    # early_stop_limit = 10
    early_stop_epochs = -1

    # train_loss_history = []

    for epoch in range(1, epoch_info.epochs+1, 1):
        model.model.train()
        model, epoch_info = epoch_training(epoch, model, train_dataloader, loss, epoch_info)
        # train_loss_history.append(epoch_info.training_loss)

        if epoch_info.save_model:
            message = f'Epoch {epoch} save model.'
            print(message)
            print(message, file=epoch_info.log)
            save_model_dir = os.path.join(result_dir, 'save_model')
            createDir(save_model_dir)
            model_path = os.path.join(save_model_dir, f'{model.model_name}-best.pkl')
            torch.save(model.model, model_path)
            model.best_model_path = model_path

        # print('Testing start.')
        # print('Testing start.', file=epoch_info.log)
        # test_file_path = os.path.join(result_dir, 'test.txt')
        # epoch_test_result_dir = os.path.join(result_dir, f'epoch_{epoch}')
        # createDir(epoch_test_result_dir)
        # testing(test_file_path, image_dir, epoch_test_result_dir, model, epoch=epoch, image_mode=image_mode, label_mode=label_mode)

        if epoch_info.epochs_no_progress > early_stop_limit:
            epoch_info.early_stop = True

        if epoch_info.early_stop and epoch <= epoch_info.epochs:
            early_stop_epochs = epoch
            message = f'Early stop! at epoch: {early_stop_epochs}'
            print(message)
            print(message, file=epoch_info.log)
            break

        model.scheduler.step()

    epoch_info.log.close()

    f = open('loss_history.txt', 'w')
    # utill = early_stop_epochs if early_stop_epochs != -1 else epoch_info.epochs
    for epoch_loss in epoch_info.train_loss_history:
        f.write(str(epoch_loss)+'\n')
    f.close()

    return model
