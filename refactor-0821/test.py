from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from common import *
from .dataset import *
from ensemble import ensemble_image
from draw import drawColor


def load_weight(path):
    return torch.load(path)


def testing(file_path, image_dir, result_dir, model, epoch=None, image_mode=ImageMode.RGB, label_mode=ImageMode.GRAY):
    dataset_pars = {
        'file_path': file_path,
        'image_dir': image_dir,
        'resize': True,
        'resize_height': 512,
        'resize_width': 512,
        'image_mode': image_mode,
    }
    test_dateset = PredictDataset(**dataset_pars)

    test_loader_pars = {
        'dataset': test_dateset,
        'batch_size': 1,
        'shuffle': False
    }
    test_dataloader = DataLoader(**test_loader_pars)

    ensemble = True
    print(f'ensemble is {ensemble}')

    with torch.no_grad():
        model_list = []
        model.model = load_weight(model.best_model_path)
        model.model.to(model.device)
        model.model.eval()
        model_list.append(model.model)

        pbar = tqdm(test_dataloader)
        for _, (image, image_name) in enumerate(pbar):
            x = image.to(model.device)
            output = model.model(x)

            if ensemble:
                prediction_prob = ensemble_image(
                    x, model_list, model.device, image_mode=image_mode)
            else:
                prediction_prob = output.cpu().numpy()

            prediction = np.squeeze(prediction_prob)
            prediction = np.argmax(prediction, axis=0)

            if epoch is None:
                drawColor(prediction, result_dir, f'{image_name[0]}')
            else:
                drawColor(prediction, result_dir, f'{epoch}-{image_name[0]}')

            pbar.set_description(f"Testing image: {image_name[0]}")
            pbar.refresh()
