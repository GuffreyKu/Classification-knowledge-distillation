import os
os.environ['LRU_CACHE_CAPACITY'] = '1'
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.tools import read_img

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
model = torch.load('savemodel/model_s_v8.pt', map_location="cpu")
model = model.to(DEVICE)
# model.eval()

class ScoreCAM:
    def __init__(self, model, img_size, label_dict, output_model_layer):
        self.model = model
        self.model.eval()
        self.img_size = img_size
        self.label_dict = label_dict
        self.fmap_block = []
        self.softmax = nn.Softmax(dim=1)
        
        # Register hook for forward pass
        output_model_layer.register_forward_hook(self.__forward_hook)

    def _get_gpu_mem(self):
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

    def resize_featuremap(self, fmap):
        return F.interpolate(fmap, size=(self.img_size[1], self.img_size[0]), mode='bilinear', align_corners=False)

    def read_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, self.img_size)
        return image

    def processing_image(self, imgfile):
        
        img = self.read_image(imgfile)
        img = img/255
        # if len(img.shape) == 2:  # Grayscale image
        #     img = np.expand_dims(img, axis=-1)
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)  # Add batch dimension
        # img = img.permute(2, 0, 1)  # Convert to (C, H, W) format
        img = img.unsqueeze(0)  # Add batch dimension
        return img.to(DEVICE)

    def __forward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def generate_heatmap(self, image_path):
        image = self.processing_image(image_path)

        with torch.no_grad():
            result, _, _ = self.model(image)
        
        arg_sm = torch.argmax(result, dim=1).item()
        pred_class_name = self.label_dict[arg_sm]
        feature_map = self.fmap_block[0]
        self.fmap_block.clear()  # Clear the fmap_block to save memory

        feature_map_resized = self.resize_featuremap(feature_map)
        feature_map_resized = feature_map_resized.detach().cpu().numpy().squeeze()

        feature_map_pre = feature_map.detach().cpu().numpy().squeeze()

        feature_map_normalized_list = [(fm - fm.min()) / (fm.max() - fm.min() + 1e-5) for fm in feature_map_resized]

        masked_input_list = []
        masked_input = image.detach().cpu().numpy()
        masked_input = np.squeeze(masked_input, axis=0)

        for act_map_normalized in feature_map_normalized_list:
            for k in range(masked_input.shape[0]):
                masked_input[k, :, :] *= act_map_normalized
            masked_input_list.append(masked_input)

        masked_input_array = torch.from_numpy(np.array(masked_input_list)).to(DEVICE)

        pred_from_masked = []
        with torch.no_grad():
            for masked_input in masked_input_array:
                pred = self.model(masked_input.unsqueeze(0))[0].detach().cpu().numpy()
                pred_from_masked.append(pred)

        pred_from_masked = np.squeeze(np.array(pred_from_masked), axis=1)

        weights = pred_from_masked[:, arg_sm]

        cam = np.dot(feature_map_pre.transpose(1, 2, 0), weights)

        heatmap = np.maximum(cam, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)
        
        del image, feature_map, feature_map_resized, feature_map_pre, masked_input_array, pred_from_masked, weights, cam
        torch.cuda.empty_cache()
        gc.collect()

        return heatmap, pred_class_name

    def plot_heatmap(self, i, image_path):
        raw = self.read_image(image_path)

        # img = cv2.resize(img, self.img_size)
        
        heatmap, pred_class_name = self.generate_heatmap(image_path)
        
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, self.img_size)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(raw, cmap='gray')
        plt.title("Raw Image")

        plt.subplot(1, 3, 2)
        plt.imshow(raw, alpha=0.6)
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.title(f"Predict Result = {pred_class_name}")

        plt.subplot(1, 3, 3)
        plt.imshow(heatmap, cmap='jet')
        plt.title("Attention map")

        plt.savefig(f"data/cam/{i}.png")
        plt.close()

label_dict = {
    "good": 0,
    "bad": 1
}

if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("data/test.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    
    paths = data['path'].to_list()

    gt = [label_dict[key] for key in data['label']]

    with torch.no_grad():
        for i, path in enumerate(paths):
            print(path)
            if (gt[i] == 1) :

                image_name = path.split('/')[-1]

                scorecam = ScoreCAM(model=model, 
                                    img_size=(230, 150), 
                                    label_dict={0: 'good', 1: 'bad'}, 
                                    output_model_layer=model.neck_f)

                scorecam.plot_heatmap(i, path)
                gc.collect()
                torch.cuda.empty_cache()