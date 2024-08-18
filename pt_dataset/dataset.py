import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch.utils.data.dataset import Dataset
import random
import cv2
import numpy as np
import pandas as pd
from random import sample
from imgaug import augmenters as iaa
from collections import Counter


def suppress_highlights_log(image):
    # Convert the image to float32
    img_float = image.astype(np.float32)
    
    # Apply logarithmic transformation
    c = 255 / np.log(1 + 255)
    log_image = np.log(img_float + 1) * c
    
    # Convert back to uint8
    log_image = np.clip(log_image, 0, 255)
    
    return log_image/np.max(log_image)

class ImgAugTransform:
    def __init__(self):
        self.aug_shift= iaa.PadToFixedSize(width=100, height=100)
        self.aug_brightness = iaa.Add((-5, 15))
        self.aug_blur = iaa.GaussianBlur(sigma=(0.1, 1.0))
        self.aug_fliplr = iaa.Fliplr(1.0)
        self.aug_affline_rot = iaa.Affine(scale={"x": (0.5, 0.9), "y": (0.5, 0.9)})
        self.aug_crop = iaa.CropAndPad(percent=(-0.25, 0.25))

    def __call__(self, img, label):
        aug = random.randint(0, 5)
        if aug == 0:
            img = img
        if aug == 1:
            img = self.aug_brightness(image=img)
        if aug == 2:
            img = self.aug_blur(image=img)
        if aug == 3:
            img = self.aug_fliplr(image=img)
        if aug == 4:
            img = self.aug_affline_rot(image=img)
        if aug == 5:
            img = self.aug_shift(image=img)
        return  img, label
  
class ImgDatasetHigh(Dataset):
    def __init__(self, mata_path, image_size:tuple, mode:str, isaug=False, balance=True):
        super().__init__()

        mata_data = pd.read_csv(mata_path)
        self.isaug = isaug
        self.labels = mata_data["label"].to_list()
        self.images_path = mata_data["path"].to_list()

        self.neg_image = []
        self.pos_image = []

        self.label_dict = {
            'bad' : 1,
            'good' : 0
        }

        if isaug:
            self.aug_fn = ImgAugTransform()

        self.balance = balance
        self.image_size = image_size
        element_counts = Counter(self.labels)

        print('-----------%s------------'%mode)
        print('number of Image Data :', len(self.images_path))
        print('number of Label Data :', len(self.labels))
        print('number of Each Label :', element_counts)

        self.read_all()

    def data_prep(self, input, mode="div"):
        if mode == "div":
            input = input/255
            input = input.astype(np.float32)
            return input
        else:
            return suppress_highlights_log(input)
        
    def read_all(self):
        for index, path in enumerate(self.images_path):
            for _ in range(1):
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                if self.isaug:
                    image = self.data_prep(image, "div")
                    image, label = self.aug_fn(image, self.label_dict[self.labels[index]])
                else:
                    image = self.data_prep(image, "div")
                    label = self.label_dict[self.labels[index]]

                image = cv2.resize(image, self.image_size)
            
                image = torch.from_numpy(image)
                image = image.unsqueeze(0)#for channel

                if label :
                    self.neg_image.append(image)
                else:
                    self.pos_image.append(image)

    def balance_sampler(self):
        self.all_image = []
        self.all_label = []

        num_neg = len(self.neg_image)
        num_pos = len(self.pos_image)

        if num_neg > num_pos:
            neg_images = sample(self.neg_image, num_pos)

            self.all_image += neg_images
            self.all_image += self.pos_image
            self.all_label += [1 for _ in range(num_pos)]
            self.all_label += [0 for _ in range(num_pos)]
        else:
            pos_images = sample(self.pos_image, num_neg)

            self.all_image += pos_images
            self.all_image += self.neg_image
            self.all_label += [0 for _ in range(num_neg)]
            self.all_label += [1 for _ in range(num_neg)]

        combined = list(zip(self.all_image, self.all_label))
        random.shuffle(combined)
        self.all_image, self.all_label = zip(*combined)

    def all_sampler(self):
        self.all_image = []
        self.all_label = []

        self.all_image += self.pos_image
        self.all_image += self.neg_image
        self.all_label += [0 for _ in range(len(self.pos_image))]
        self.all_label += [1 for _ in range(len(self.neg_image))]

    def __getitem__(self, index):
        image = self.all_image[index]
        target = self.all_label[index]

        if (index == len(self.all_label)-1) and (self.balance):
            self.balance_sampler()
        return image, target

    def __len__(self):
        return len(self.all_label)