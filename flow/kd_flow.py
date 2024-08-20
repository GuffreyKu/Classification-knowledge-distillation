import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def calculate_accuracy(logits, target):
    logits = torch.softmax(logits, dim=-1).argmax(dim=-1)
    n = logits.size(0)
    correct = logits.eq(target.data).cpu().sum()
    accuracy = correct/float(n)
    return accuracy

def class_weight(target, num_cls=2):
    target=target.numpy(force=True)
    unique, counts = np.unique(np.array(target), return_counts=True)
    num_class = dict(zip(unique, counts))
    alpha = np.ones((num_cls,))

    for c in num_class.keys():
        w = target.shape[0] / (num_class[c] * len(num_class))
        alpha[ c ] = w

    alpha = alpha.astype(np.float32)
    return torch.from_numpy(alpha)

def train(now_ep,
          teacherNet,
          studentNet,
          optimizer,
          scheduler,
          dataloader,
          criterion,
          scaler,
          DEVICE):
    
    losses = []
    matrixes = []
    teacherNet.eval()
    studentNet.train()
    with tqdm(dataloader, ascii=' =', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as loader:
        for image, target in loader:
            loader.set_description(f"train {now_ep}")
            image = image.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits, teacher_medium, teacher_deep= teacherNet(image)
                teacher_embed = {
                    "medium":teacher_medium,
                    "deep":teacher_deep
                }
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred, embed_medium, embed_deep = studentNet(image)
                alpha = class_weight(target)
                embed = {
                    "medium":embed_medium,
                    "deep":embed_deep
                }
                loss = criterion(teacher_logits, pred, teacher_embed, embed, target, alpha.to(DEVICE))
            
            lr = get_lr(optimizer)

            matrix = calculate_accuracy(pred, target)

            matrixes.append(matrix)
            losses.append(loss.item())

            scheduler.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loader.set_postfix(loss=np.mean(losses), matrix=np.mean(matrixes), lr=lr)
    return np.mean(losses), np.mean(matrixes)


def evaluate(mode,
             model,
             dataloader,
             criterion,
             DEVICE):
    
    model.eval()
    losses = []
    matrixes = []
    with torch.no_grad():
        with tqdm(dataloader, ascii=' =', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as loader:
            for image, target in loader:
                loader.set_description(f"{mode}")
                image = image.to(DEVICE)
                target = target.to(DEVICE)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred, _, _ = model(image)
                    alpha = class_weight(target)
                    loss = criterion(pred, target, alpha.to(DEVICE))

                matrix = calculate_accuracy(pred, target)

                matrixes.append(matrix)
                losses.append(loss.item())

                loader.set_postfix(loss=np.mean(losses), matrix=np.mean(matrixes))
    return np.mean(losses), np.mean(matrixes)