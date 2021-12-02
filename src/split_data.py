import os
import random

from common import *

def split_data(mode, result_dir, image_dir, label_dir=None, k=None, valid=False, test=True):
    """
        * param mode : select train or predict
        * param result_dir : the directory that save the dataset file
        * param image_dir : the directory that stroe the images
        * param label_dir : the directory that stroe the labels, when is None, labels name is same as images name
        * param k : the split number of k fold cross validation, when is None, will split into single fold
        * param valid : use valid dataset or not, default is False
    """
    seed = random.randint(0, 2147483647)

    image_name_list = os.listdir(image_dir)
    random.seed(seed)
    random.shuffle(image_name_list)
    if not label_dir is None:
        label_name_list = os.listdir(label_dir)
        random.seed(seed)
        random.shuffle(label_name_list)
    else:
        label_name_list = image_name_list

    data_list = list(zip(image_name_list, label_name_list))

    if mode == Mode.PREDICT:
        file_path = os.path.join(result_dir, 'predict.txt')
        assign_data(data_list, file_path)
    elif mode == Mode.TRAIN:
        assert (k is None) or (isinstance(k, int))
        if k is None or k == 1:
            assign_single_fold(5, data_list, result_dir, valid, test)
        elif k > 1:
            assign_k_fold(k, data_list, result_dir, valid)
        else:
            raise 'k value is invalid!'
    else:
        raise 'mode value is invalid!'

def assign_data(data_list, file_path):
    f = open(file_path, 'w')
    first_row = True
    for data in data_list:
        row = '-'.join(list(data))
        if first_row:
            first_row = False
        else:
            row = '\n' + row
        f.write(row)
    f.close()

def partition_data(k, data_list, result_dir):
    data_num = len(data_list)
    per_partition = int(data_num/k)
    remaining_num = data_num % k

    end_ = 0
    for i in range(1, k+1, 1):
        file_path = os.path.join(result_dir, f'partition {i}.txt')
        f = open(file_path, 'w')
        if remaining_num > 0:
            from_, end_ = end_, (end_ + per_partition + 1)
            remaining_num -= 1
        else:
            from_, end_ = end_, (end_ + per_partition)

        first_row = True
        for data in data_list[ from_:end_]:
            row = '-'.join(list(data))
            if first_row:
                first_row = False
            else:
                row = '\n' + row
            f.write(row)
        f.close

def assign_fold(k, result_dir, fold_dir=None, valid=False, test=True, test_partition=None):
    if fold_dir is None:
        fold_dir = result_dir

    partion_num = [e for e in range(1, k+1, 1)]
    
    train_path = os.path.join(fold_dir, 'train.txt')
    train_f = open(train_path, 'w')
    if valid:
        valid_path = os.path.join(fold_dir, 'valid.txt')
        valid_f = open(valid_path, 'w')
    test_path = os.path.join(fold_dir, 'test.txt')
    test_f = open(test_path, 'w')

    
    if not test_partition is None:
        test_partition = test_partition
    else:
        test_partition = random.choices(partion_num) if test else []
    for e in test_partition:
        partion_num.remove(e)

    valid_partition = random.choices(partion_num) if valid else []
    for e in valid_partition:
        partion_num.remove(e)

    train_partition = partion_num

    train_first_file = True
    
    for j in range(1, k+1, 1):
        file_path = os.path.join(result_dir, f'partition {j}.txt')
        f = open(file_path, 'r')
        if j in train_partition:
            if train_first_file:
                train_first_file = False
            else:
                train_f.write('\n')
            for line in f:
                train_f.write(line)
        elif j in valid_partition:
            for line in f:
                valid_f.write(line)
        elif j in test_partition:
            for line in f:
                test_f.write(line)
        else:
            raise 'partition error!'
        f.close()

    train_f.close()
    if valid:
        valid_f.close()
    test_f.close()

def assign_k_fold(k, data_list, result_dir, valid):
    partition_data(k, data_list, result_dir)

    for i in range(1, k+1, 1):
        fold_path = os.path.join(result_dir, f'fold {i}')
        createDir(fold_path)
        assign_fold(k, result_dir, fold_path, valid, test_partition=[i])

def only_assign_k_fold(k, result_dir, valid):
    for i in range(1, k+1, 1):
        fold_path = os.path.join(result_dir, f'fold {i}')
        createDir(fold_path)
        assign_fold(k, result_dir, fold_path, valid, test_partition=[i])

def assign_single_fold(k, data_list, result_dir, valid, test):
    partition_data(k, data_list, result_dir)

    assign_fold(k, result_dir, valid=valid, test=test)

def copy_partitions(partitions_dir, result_dir):
    import shutil
    list_ = os.listdir(partitions_dir)
    # print(list_)
    partition_files = []
    partition_counts = 0
    for file in list_:
        if file.split('.')[-1] == 'txt':
            partition_files.append(file)
            src = os.path.join(partitions_dir, file)
            dst = result_dir
            shutil.copy2(src=src, dst=dst)
            # print(src, dst)
            partition_counts += 1
    return partition_counts