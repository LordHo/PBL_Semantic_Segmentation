from tqdm import tqdm

from torch.utils.data import DataLoader

from common import *
from dataset import *
from combo_loss import ComboLoss
from UNet_Version.loss.iouLoss import IOU_loss
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
        self.epoch_combo_loss = 0.0
        self.epoch_iou_loss = 0.0

    def epoch_message(self, epoch_info, batch_num, epoch, last_lr):
        list_ = [f"Epoch {epoch:2d}"]

        print_training_loss = False
        if self.training_loss > 0.0 and print_training_loss:
            train_loss_message = f'Training loss: {epoch_info.training_loss/batch_num:.3f}'
            list_.append(train_loss_message)

        print_combo_loss = True
        if self.epoch_combo_loss != 0.0 and print_combo_loss:
            combo_loss_message = f'Combo loss: {epoch_info.epoch_combo_loss/batch_num:.3f}'
            list_.append(combo_loss_message)

        print_iou = True
        if self.epoch_iou_loss > 0.0 and print_iou:
            iou_message = f'IoU: {1 - epoch_info.epoch_iou_loss/batch_num:.3f}'
            list_.append(iou_message)

        last_lr_message = f'lr: {last_lr:.1e}'
        list_.append(last_lr_message)

        message = ', '.join(list_)
        return message


def epoch_training(epoch, model, dataloader, combo_loss_func, epoch_info, iou_loss_func):
    epoch_info.resetepochloss()
    epoch_info.reset()

    pbar = tqdm(dataloader)
    for batch_index, (batch_image, batch_label, _, _) in enumerate(pbar):
        x, y = batch_image.to(model.device), batch_label.to(model.device)
        y = y.type(torch.cuda.FloatTensor)

        model.optimizer.zero_grad()  # reset the gradient to zero
        output = model.model(x)

        combo_loss = combo_loss_func(output, y)
        iou_loss = iou_loss_func(output, y)

        loss = combo_loss + 0.0 * iou_loss
        """ Zero gradients, perform a backward pass, and update the weights. """
        loss.backward()
        model.optimizer.step()

        epoch_info.training_loss += loss.item()
        epoch_info.epoch_combo_loss += combo_loss.item()
        epoch_info.epoch_iou_loss += iou_loss.item()

        last_lr = model.scheduler.get_last_lr()[0]
        batch_num = batch_index+1

        message = epoch_info.epoch_message(
            epoch_info, batch_num, epoch, last_lr)

        pbar.set_description(message)
        pbar.refresh()
    print(message, file=epoch_info.log)

    if epoch_info.best_iou_loss > epoch_info.epoch_iou_loss:
        epoch_info.best_iou_loss = epoch_info.epoch_iou_loss
        epoch_info.epochs_no_progress = 0
        epoch_info.save_model = True
    else:
        epoch_info.epochs_no_progress += 1

    epoch_info.train_loss_history.append(epoch_info.training_loss)

    return model, epoch_info


def training(file_path, image_dir, label_dir, result_dir, model, image_mode, label_mode):

    print(f'Device: {model.device}')

    dataset_pars = {
        'file_path': file_path,
        'image_dir': image_dir,
        'label_dir': label_dir,
        'resize': True,
        'resize_height': 512,
        'resize_width': 512,
        'repeat': 20,
        'n_classes': 10+1,
        'trans': True,
        'image_mode': image_mode,
        'label_mode': label_mode,
    }
    train_dateset = TrainDataset(**dataset_pars)

    train_loader_pars = {
        'dataset': train_dateset,
        'batch_size': 3,
        'shuffle': False
    }
    train_dataloader = DataLoader(**train_loader_pars)

    epoch_info = EpochInfo(setting_epochs=100)

    repeat = dataset_pars['repeat']
    batch_size = train_loader_pars['batch_size']
    image_shape = (dataset_pars['resize_height'], dataset_pars['resize_width'])
    message = f'Training epochs: {epoch_info.epochs}, repeat: {repeat}, batch size: {batch_size}, image shape (height, width): {image_shape}'
    print(message)

    lof_path = os.path.join(result_dir, 'log.txt')
    epoch_info.log = open(lof_path, 'w')
    print(message, file=epoch_info.log)

    print(f'Used losses: combo loss.')
    print(f'Used losses: combo loss.', file=epoch_info.log)

    early_stop_limit = epoch_info.epochs
    # early_stop_limit = 10
    early_stop_epochs = -1

    # train_loss_history = []

    combo_loss = ComboLoss()
    iou_loss = IOU_loss

    for epoch in range(1, epoch_info.epochs+1, 1):
        model.model.train()
        model, epoch_info = epoch_training(
            epoch, model, train_dataloader, combo_loss, epoch_info, iou_loss)
        # train_loss_history.append(epoch_info.training_loss)

        if epoch_info.save_model:
            message = f'Epoch {epoch} save model.'
            print(message)
            print(message, file=epoch_info.log)
            save_model_dir = os.path.join(result_dir, 'save_model')
            createDir(save_model_dir)
            model_path = os.path.join(
                save_model_dir, f'{model.model_name}-best.pkl')
            torch.save(model.model, model_path)
            model.best_model_path = model_path

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
