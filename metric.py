
from pt_dataset.dataset import ImgDatasetHigh

import numpy as np
import torch

from sklearn import metrics

DEVICE = torch.device("cpu")

model = torch.jit.load('savemodel/model_s_v1_trace.pt', map_location="cpu")
model = model.to(DEVICE)

target_names = ['normal', 'abnormal']
image_size = (224, 224)

if __name__ == '__main__':

    train_dataset = ImgDatasetHigh("data/train.csv", image_size=image_size, mode="train")
    valid_dataset = ImgDatasetHigh("data/test.csv", image_size=image_size, mode="test")

    train_dataset.all_sampler()
    valid_dataset.all_sampler()
    
    # torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=1))
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    data_loader_valid = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )


    train_pred = []
    valid_pred = []

    train_gt = []
    valid_gt = []

    model.eval()
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader_train):
            image, target= image.to(DEVICE), target.to(DEVICE)
            output, _, _ = model(image)
            output = torch.argmax(output, dim=-1)
            output = output.view(-1).numpy(force=True)
            target = target.view(-1).numpy(force=True)

            train_pred.append(output)
            train_gt.append(target)

        train_pred = np.concatenate(train_pred)
        train_gt = np.concatenate(train_gt)

        print("------------ Train ---------------")
        print(metrics.classification_report(train_gt, train_pred, target_names=target_names))

        del train_dataset, data_loader_train

        for i, (image, target) in enumerate(data_loader_valid):
            image, target= image.to(DEVICE), target.to(DEVICE)
            output, _, _ = model(image)
            output = torch.argmax(output, dim=-1)
            output = output.view(-1).numpy(force=True)
            target = target.view(-1).numpy(force=True)

            valid_pred.append(output)
            valid_gt.append(target)

        valid_pred = np.concatenate(valid_pred)
        valid_gt = np.concatenate(valid_gt)

        print("------------ Test ---------------")
        print(metrics.classification_report(valid_gt, valid_pred, target_names=target_names))

        del valid_dataset, data_loader_valid
