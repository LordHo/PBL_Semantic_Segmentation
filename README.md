# PBL_Segmentation_20211102

## Installation
```
pip install -r requirements.txt
```

## Train

```
# in cmd
cd src
python main.py
```

### Parameters

#### In `main.py`
```
# at line 16
mode = Mode.TRAIN
image_mode = ImageMode.GRAY
label_mode = ImageMode.GRAY
# if use_existes_partition is True, will use the existed parititions as training partitions.
use_existes_partition = True

# at line 24
# If useTEST is True, after training, will pred testing images.
useTEST = True

# at line 28
result_dir = 'The dir you want to save result'

# at line 33
train_root_dir = 'The training images dir'

# at line39
label_dir = 'The training lables dir'

# at line 53
partitions_dir = 'The existed parititions dir'

# at line 67
# This is the model setting
train_pars = {
    # 10 foreground class and 1 background(nothing, for 
    augmentation).
    'n_classes'     : 10 + 1,
    'model_name'    : 'UNet',
    'backbone'      : 'efficientnet-b3',
    'activation'    : 'softmax2d',
    'image_mode'    : image_mode,
    }

# at line 143
predict_file_path = 'The predict images list file'
```

#### In `train.py`
```
# at line 151
# modified the resize height and width to 1024 when training by 1024*1024.
dataset_pars = {
    'file_path': file_path,
    'image_dir': image_dir,
    'label_dir': label_dir,
    'resize': True,
    'resize_height': 512,
    'resize_width': 512,
    'repeat': 20,
    'n_classes': 10 + 1,
    'trans': True,
    'image_mode': image_mode,
    'label_mode': label_mode,
}

# at line 167
# modified the batch_size as the cpu and gpu can afford.
train_loader_pars = {
    'dataset': train_dateset,
    'batch_size': 3,
    'shuffle': False
}

# at line 177
# modified setting_epochs as the epochs that required.
epoch_info = EpochInfo(setting_epochs=100)
```
 
## Predict
```
# in cmd
cd src
python main.py
```

#### In `main.py`
```
# at line 16
mode = Mode.PREDICT
# Need to match the loaded model in following
image_mode = ImageMode.GRAY
label_mode = ImageMode.GRAY
# if use_existes_partition is True, will use the existed parititions as training partitions.
# use_existes_partition only work in train mode.
use_existes_partition = 'Not work'

# at line 24
# useTEST only work in train mode.
# If useTEST is True, after training, will pred testing images.
useTEST = 'Not work'

# at line 28
result_dir = 'The dir you want to save result'

# at line 33
# also use in predict mode, but is for predict images
train_root_dir = 'The training images dir'

# at line39
# label_dir only work in train mode.
label_dir = 'Not work'

# at line 53
# partitions_dir only work in train mode.
partitions_dir = 'Not work'

# at line 143
predict_file_path = 'The predict images list file'
```

#### In `predict.py`
```
# at line 16
# modified the resize height and width to 1024 when predicting by 1024*1024.
dataset_pars = {
    'file_path': file_path,
    'image_dir': image_dir,
    'resize': True,
    'resize_height': 512,
    'resize_width': 512,
    'image_mode': image_mode,
}
```

