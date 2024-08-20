import torch
from torch.utils.data.distributed import DistributedSampler
from pt_dataset.dataset import ImgDataset, ImgDatasetHigh

def dataloader(train_path, valid_path, batch_size, image_size):
    train_dataset = ImgDatasetHigh(train_path, image_size=image_size, mode="train", isaug=True)
    valid_dataset = ImgDatasetHigh(valid_path, image_size=image_size, mode="test", isaug=False)
    
    train_dataset.balance_sampler()
    valid_dataset.balance_sampler()
    # train_dataset.all_sampler()
    # valid_dataset.all_sampler()

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset)


    trainLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True
    )

    vaildLoader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True
    )

    return trainLoader, vaildLoader, train_sampler, valid_sampler