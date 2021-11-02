import os
import time

from common import *
from split_data import split_data, only_assign_k_fold, copy_partitions
from train import training
from test import testing
from predict import predicting

if __name__ == '__main__':
    mode = Mode.TRAIN
    image_mode = ImageMode.RGB
    label_mode = ImageMode.GRAY
    use_existes_partition = True
    """
        TEST only when after TRAIN, if only want TEST, use PREDICT instead
    """
    useTEST = True
    datasets = [Dataset.TRAIN, Dataset.TEST] if useTEST else [Dataset.TRAIN]

    timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    result_dir = os.path.join('..', 'result', timestamp)
    createDir(result_dir)

    train_root_dir = os.path.join('..', 'dataset', 'modify_single_train')
    # image_dir = os.path.join('..', 'dataset', '14X label', 'can add into labels ( stage 3 in it)', 'selection')
    # image_dir = os.path.join('..', 'dataset', 'pre_train', 'images')
    # image_dir = os.path.join('..', 'dataset', 'root_and_image_data', 'images')
    # image_dir =  os.path.join('..', '..', 'segmentation', 'image')
    image_dir = os.path.join(train_root_dir, 'images')
    print(f'Image Dir: {image_dir}')

    if mode == Mode.TRAIN:
        k = 3
        # label_dir = os.path.join('..', 'dataset', 'pre_train', 'labels')
        label_dir = os.path.join(train_root_dir, 'labels')
        print(f'Label Dir: {label_dir}')

        valid = True if Dataset.VALID in datasets else False
        test = True if Dataset.TEST in datasets else False
        " test only when single dir can remove, if k fold will at least a pratition be testing data "
        if not use_existes_partition:
            split_data(mode, result_dir, image_dir, label_dir=label_dir, k=k, valid=valid, test=test)
        else:
            # partitions_dir = os.path.join('..', 'special partitions', 'Single_Row_132_image')
            partitions_dir = os.path.join('..', 'special partitions', 'Single_Patient_Single_Row_132_image')
            print(f'Special Paritions: {partitions_dir}')
            partition_counts = copy_partitions(partitions_dir, result_dir)
            if partition_counts == 0:
                raise "Partition Error!"
            else:
                k = partition_counts
            only_assign_k_fold(k, result_dir, valid=valid)
        print(f'Total Folder Number: {k}')
    
        train_pars = {
            'n_classes'     : 10 + 1,
            # 'model_name'    : 'UNet', 
            'model_name'    : 'PSPNet',
            # 'backbone'      : 'efficientnet-b3',
            'backbone'      : 'resnet34',
            'activation'    : 'softmax2d',
            'image_mode'    : image_mode,
        }
        # image_mode options are 'rgb', 'gray'

        if k == 1 or k is None:
            model = Model(**train_pars)
            if Dataset.TRAIN in datasets:
                train_file_path = os.path.join(result_dir, 'train.txt')
                print(f'Training.')
                model = training(train_file_path, image_dir, label_dir, result_dir, model, image_mode=image_mode, label_mode=label_mode)

                if Dataset.TEST in datasets:
                    test_file_path = os.path.join(result_dir, 'test.txt')
                    testing(test_file_path, image_dir, result_dir, model, image_mode=image_mode, label_mode=label_mode)
        elif k > 1:
            for i in range(1, k+1, 1):
                model = Model(**train_pars)
                fold_dir = os.path.join(result_dir, f'fold {i}')

                if Dataset.TRAIN in datasets:
                    train_file_path = os.path.join(fold_dir, 'train.txt')
                    print(f'Fold {i} is training.')
                    model = training(train_file_path, image_dir, label_dir, fold_dir, model, image_mode=image_mode, label_mode=label_mode)

                    if Dataset.TEST in datasets:
                        test_file_path = os.path.join(fold_dir, 'test.txt')
                        testing(test_file_path, image_dir, fold_dir, model, image_mode=image_mode, label_mode=label_mode)
        else:
            raise "k value is wrong"

    elif mode == Mode.PREDICT:
        fold_name = '2021-10-07 16-09-05-copy(JingPing+BCE)'
        split_data(mode, result_dir, image_dir)

        predict_file_path = os.path.join(result_dir, 'predict.txt')
        load_weight_path_list = [
           os.path.join('..', 'result', fold_name, 'fold 1', 'save_model', 'UNet-best.pkl'),
           os.path.join('..', 'result', fold_name, 'fold 2', 'save_model', 'UNet-best.pkl'),
           os.path.join('..', 'result', fold_name, 'fold 3', 'save_model', 'UNet-best.pkl'),
           os.path.join('..', 'result', fold_name, 'fold 4', 'save_model', 'UNet-best.pkl'),
           os.path.join('..', 'result', fold_name, 'fold 5', 'save_model', 'UNet-best.pkl'),
        ]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicting(predict_file_path, image_dir, result_dir, load_weight_path_list, device, image_mode)
    
    else:
        raise "Mode is wrong."