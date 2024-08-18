import os
import torch
import cv2
import numpy as np

def folderCheck(folders:list):
    for path in folders:
        if not os.path.exists(path):
            os.mkdir(path)


def traced_func(model, saved_path, X):
    traced_model = torch.jit.trace(model, X)
    torch.jit.save(traced_model, saved_path)
    return traced_model

def suppress_highlights_log(image):
    # Convert the image to float32
    img_float = image.astype(np.float32)
    
    # Apply logarithmic transformation
    # c = 255 / np.log(1 + np.max(img_float))
    log_image = (np.log(img_float + 1))
    
    # Convert back to uint8
    log_image = np.clip(log_image, 0, 255)
    
    return log_image/np.max(log_image)

def read_img(path, image_size, DEVICE):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raw = image
    image = cv2.resize(image, image_size)
    # image = suppress_highlights_log(image)
    # raw = image*255
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)#for channel
    image = image.unsqueeze(0)#for channel

    image = image/255
    image = image.to(DEVICE)
    return image, raw