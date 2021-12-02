from tqdm import tqdm

from torch.utils.data import DataLoader
import numpy as np
from common import *
from dataset import *
from ensemble import ensemble_image
from draw import drawColor


def load_weight(path):
    return torch.load(path)


def predicting(file_path, image_dir, result_dir, load_weight_path_list=None, device='cpu', image_mode=ImageMode.RGB):
    dataset_pars = {
        'file_path': file_path,
        'image_dir': image_dir,
        'resize': True,
        'resize_height': 512,
        'resize_width': 512,
        'image_mode': image_mode,
    }
    predict_dateset = PredictDataset(**dataset_pars)

    predict_loader_pars = {
        'dataset': predict_dateset,
        'batch_size': 1,
        'shuffle': False
    }
    predict_dataloader = DataLoader(**predict_loader_pars)

    ensemble = True
    print(f'ensemble is {ensemble}')

    with torch.no_grad():

        model_list = []
        for load_weight_path in load_weight_path_list:
            model = load_weight(load_weight_path)
            model.to(device)
            model.eval()
            model_list.append(model)

        pbar = tqdm(predict_dataloader)
        for index, (image, image_name) in enumerate(pbar):
            x = image.to(device)

            output = model(x)

            if ensemble:
                prediction_prob = ensemble_image(
                    x, model_list, device, image_mode=image_mode)
            else:
                prediction_prob = output.cpu().numpy()

            prediction = np.squeeze(prediction_prob)
            prediction = np.argmax(prediction, axis=0)
            drawColor(prediction, result_dir, image_name[0])

            pbar.set_description(f"Predicting image: {image_name[0]}")
            pbar.refresh()
