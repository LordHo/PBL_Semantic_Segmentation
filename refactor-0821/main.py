import os
import time
import glob

from common import *
from split_data import split_data, only_assign_k_fold, copy_partitions
from train import training
from .test import testing
from predict import predicting


def main_log(message, file):
    file.write(message+'\n')


if __name__ == '__main__':
    mode = Mode.PREDICT
    image_mode = ImageMode.RGB
    # image_mode = ImageMode.GRAY
    label_mode = ImageMode.GRAY
    use_existes_partition = True
    use_special_predict_list = False
    """
        TEST only when after TRAIN, if only want TEST, use PREDICT instead
    """
    useTEST = True
    datasets = [Dataset.TRAIN, Dataset.TEST] if useTEST else [Dataset.TRAIN]

    timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    result_dir = os.path.join('..', 'result', timestamp)
    createDir(result_dir)

    mainLog_file = open(os.path.join(result_dir, 'main log.txt'), 'w')

    train_root_dir = os.path.join('..', 'dataset', 'modify_single_train')
    image_dir = os.path.join(train_root_dir, 'images')

    pred_image_dir = os.path.join('..', 'dataset', '2511')
    image_dir = pred_image_dir

    print(f'Image Dir: {image_dir}')
    main_log(f'Image Dir: {image_dir}', mainLog_file)

    if mode == Mode.TRAIN:
        k = 3
        label_dir = os.path.join(train_root_dir, 'labels')
        print(f'Label Dir: {label_dir}')
        main_log(f'Label Dir: {label_dir}', mainLog_file)

        valid = True if Dataset.VALID in datasets else False
        test = True if Dataset.TEST in datasets else False
        " test only when single dir can remove, if k fold will at least a pratition be testing data "
        if not use_existes_partition:
            split_data(mode, result_dir, image_dir,
                       label_dir=label_dir, k=k, valid=valid, test=test)
        else:
            partitions_dir = os.path.join(
                '..', 'special partitions', 'Single_Patient_Single_Row_132_image')
            print(f'Special Paritions: {partitions_dir}')
            main_log(f'Special Paritions: {partitions_dir}', mainLog_file)
            partition_counts = copy_partitions(partitions_dir, result_dir)
            if partition_counts == 0:
                raise "Partition Error!"
            else:
                k = partition_counts
            only_assign_k_fold(k, result_dir, valid=valid)
        print(f'Total Folder Number: {k}')
        main_log(f'Total Folder Number: {k}', mainLog_file)

        # U-Net Model
        # train_pars = {
        #     'n_classes'     : 10 + 1,
        #     'model_name'    : 'UNet',
        #     'backbone'      : 'efficientnet-b3',
        #     'activation'    : 'softmax2d',
        #     'image_mode'    : image_mode,
        # }

        # PSPNet Model
        # train_pars = {
        #     'n_classes'     : 10 + 1,
        #     'model_name'    : 'PSPNet',
        #     'backbone'      : 'resnet152',
        #     'activation'    : 'softmax2d',
        #     'image_mode'    : image_mode,
        # }

        # DeepLab v3+ Model
        # train_pars = {
        #     'n_classes': 10 + 1,
        #     'model_name': 'DeepLabv3+',
        #     'backbone': 'efficientnet-b3',
        #     'activation': 'softmax2d',
        #     'image_mode': image_mode,
        # }

        # UNet++ Model
        train_pars = {
            'n_classes': 10 + 1,
            'model_name': 'UNet++',
            'backbone': 'resnext50_32x4d',
            'activation': 'softmax2d',
            'image_mode': image_mode,
        }

        main_log(f'Train pars:', mainLog_file)
        for key in train_pars.keys():
            main_log(f'{key}: {train_pars[key]}', mainLog_file)
        # image_mode options are 'rgb', 'gray'
        print(train_pars)
        if k == 1 or k is None:
            model = Model(**train_pars)
            if Dataset.TRAIN in datasets:
                train_file_path = os.path.join(result_dir, 'train.txt')
                print(f'Training.')
                model = training(train_file_path, image_dir, label_dir, result_dir,
                                 model, image_mode=image_mode, label_mode=label_mode)

                if Dataset.TEST in datasets:
                    test_file_path = os.path.join(result_dir, 'test.txt')
                    testing(test_file_path, image_dir, result_dir, model,
                            image_mode=image_mode, label_mode=label_mode)
        elif k > 1:
            for i in range(1, k+1, 1):
                model = Model(**train_pars)
                fold_dir = os.path.join(result_dir, f'fold {i}')

                if Dataset.TRAIN in datasets:
                    train_file_path = os.path.join(fold_dir, 'train.txt')
                    print(f'Fold {i} is training.')
                    model = training(train_file_path, image_dir, label_dir, fold_dir,
                                     model, image_mode=image_mode, label_mode=label_mode)

                    if Dataset.TEST in datasets:
                        test_file_path = os.path.join(fold_dir, 'test.txt')
                        testing(test_file_path, image_dir, fold_dir, model,
                                image_mode=image_mode, label_mode=label_mode)
        else:
            raise "k value is wrong"

    elif mode == Mode.PREDICT:
        # DeepLabv3+
        # fold_name = '2021-11-14 13-33-17-DeepLabv3+'

        # Unet
        fold_name = '2021-10-27 19-29-14-JingPing+BCE-OnlySingle(132)-fold(9)'

        if use_special_predict_list:
            predict_file_path = os.path.join(
                '..', 'result', '2021-11-17 22-14-06', 'partition 1.txt')
        else:
            split_data(mode, result_dir, image_dir)
            predict_file_path = os.path.join(result_dir, 'predict.txt')

        # load_weight_path_list = [
        #     os.path.join('..', 'result', fold_name, 'fold 1',
        #                  'save_model', 'UNet++-best.pkl'),
        # ]

        load_weight_path_list = []
        for i in range(1, 9+1):

            model_path = os.path.join(
                '..', 'result', fold_name, f'fold {i}', 'save_model', '*.pkl')

            model_path = glob.glob(model_path)[0]
            load_weight_path_list.append(model_path)
            print(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicting(predict_file_path, image_dir, result_dir,
                   load_weight_path_list, device, image_mode)

    else:
        raise "Mode is wrong."

    mainLog_file.close()
