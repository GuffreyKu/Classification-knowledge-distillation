import torch
from pt_dataset.dataset import ImgDataset, ImgDatasetHigh

def dataloader(train_path, valid_path, batch_size, image_size):
    train_dataset = ImgDatasetHigh(train_path, image_size=image_size, mode="train", isaug=True)
    valid_dataset = ImgDatasetHigh(valid_path, image_size=image_size, mode="test", isaug=False)
    
    train_dataset.balance_sampler()
    valid_dataset.balance_sampler()
    # train_dataset.all_sampler()
    # valid_dataset.all_sampler()
    trainLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True
    )

    vaildLoader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True
    )

    return trainLoader, vaildLoader