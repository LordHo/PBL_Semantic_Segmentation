import cv2
import time
import numpy as np

import torch

stop_time = False
stop_time_sec = 0.1

def ensemble_image(img, model_list, device, flip_=True):
    # print(f'img type: {type(img)}, img shape: {img.shape}')
    ensemble_final_result = None
    # print(f'ensemble_final_result type: {type(ensemble_final_result)}')
    for index, model in enumerate(model_list):
        # pr_img = model.predict(img) # keras
        pr_img = model(img) # pytorch
        if stop_time:
            time.sleep(stop_time_sec)
        # print(f'pr_img type: {type(pr_img)}')
        pr_img = pr_img.cpu().numpy()
        ensemble_img = pr_img.copy()
        if ensemble_final_result is None:
            ensemble_final_result = np.zeros(ensemble_img.shape, dtype=np.float64)

        if flip_:
            """ from tensor to numpy from device to cpu """
            img_ = img.cpu().numpy()
            imgp = np.squeeze(img_)

            """ due to tensor is [c, h, w], but cv2 need [h, w, c] """
            imgp = imgp.transpose(1, 2, 0)
            """ Horizontally flip """
            h_flip = cv2.flip(imgp, 1)
            """ [h, w, c] -> [c, h, w] """
            h_flip = h_flip.transpose(2, 0, 1)
            h_flip = np.expand_dims(h_flip, axis=(0))
            """ from numpy to tensor and also from cpu to device(which is model location) """
            h_flip_ = torch.tensor(h_flip)
            h_flip_ = h_flip_.to(device)
            pre_h_flip = model(h_flip_)
            if stop_time:
                time.sleep(stop_time_sec)
            """ from tensor to numpy from device to cpu """
            pre_h_flip_ = pre_h_flip.cpu().numpy()
            pre_h_flip_ = np.squeeze(pre_h_flip_)

            pre_h_flip_ = pre_h_flip_.transpose(1, 2, 0)
            pre_h_flip_ = cv2.flip(pre_h_flip_, 1)
            h_flip = pre_h_flip_.transpose(2, 0, 1)
            # h_flip = pre_h_flip_
            
            """ from tensor to numpy from device to cpu """
            v_flip = cv2.flip(imgp, 0)
            v_flip = v_flip.transpose(2, 0, 1)
            v_flip = np.expand_dims(v_flip, axis=(0))
            """ from numpy to tensor and also from cpu to device """
            v_flip_ = torch.tensor(v_flip)
            v_flip_ = v_flip_.to(device)
            pre_v_flip = model(v_flip_)
            if stop_time:
                time.sleep(stop_time_sec)
            """ from tensor to numpy from device to cpu """
            pre_v_flip_ = pre_v_flip.cpu().numpy()
            pre_v_flip_ = np.squeeze(pre_v_flip_)

            pre_v_flip_ = pre_v_flip_.transpose(1, 2, 0)
            pre_v_flip_ = cv2.flip(pre_v_flip_, 0)
            v_flip = pre_v_flip_.transpose(2, 0, 1)
            # v_flip = pre_v_flip_
            
            
            """ from tensor to numpy from device to cpu """
            hv_flip = cv2.flip(imgp, -1)
            hv_flip = hv_flip.transpose(2, 0, 1)
            hv_flip = np.expand_dims(hv_flip, axis=(0))
            """ from numpy to tensor and also from cpu to device """
            hv_flip_ = torch.tensor(hv_flip)
            hv_flip_ = hv_flip_.to(device)
            pre_hv_flip = model(hv_flip_)
            if stop_time:
                time.sleep(stop_time_sec)
            """ from tensor to numpy from device to cpu """
            pre_hv_flip_ = pre_hv_flip.cpu().numpy()
            pre_hv_flip_ = np.squeeze(pre_hv_flip_)

            pre_hv_flip_ = pre_hv_flip_.transpose(1, 2, 0)
            pre_hv_flip_ = cv2.flip(pre_hv_flip_, -1)
            hv_flip = pre_hv_flip_.transpose(2, 0, 1)
            # hv_flip = pre_hv_flip_
            
    # divide first
            ensemble_img = (pr_img + h_flip + v_flip + hv_flip)/4
        ensemble_final_result = ensemble_final_result + ensemble_img
    ensemble_final_result = ensemble_final_result/(index + 1)
    # divide after
            # print(pr_img.shape, h_flip.shape, v_flip.shape, hv_flip.shape)
    #         ensemble_img = (pr_img + h_flip + v_flip + hv_flip)
    #     ensemble_final_result = ensemble_final_result + ensemble_img
    # ensemble_final_result = ensemble_final_result/(index + 1)/4

    return ensemble_final_result