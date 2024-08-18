import torch
import gc 
import torch_optimizer as optim_alg
from pt_dataset.dataloader import dataloader
from model.model import Efficientnet_s, MobileNet, MaxVit, MobileNetL, Resnet

from flow.kd_flow import train, evaluate
from utils.pytorchtools import EarlyStopping, CosineDecayWarmup
from utils.tools import folderCheck, traced_func
from model.loss import KdCeLoss, CeLoss
from utils.loss_vis import drow

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

epochs = 150
batch_size = 32

image_size = (224, 224)
model_path = "savemodel"
model_verison = "kd_v1"
trainLoader, vaildLoader = dataloader(train_path="data/train.csv",
                                    valid_path="data/test.csv",
                                    batch_size=batch_size,
                                    image_size=image_size)

# student_model = Efficientnet_s(2).to(DEVICE)
student_model = MobileNet(2).to(DEVICE)
# student_model = mobilevit_xxs(2).to(DEVICE)

# model = Efficientnet(2).to(DEVICE)
# model = mobilevit_s(2).to(DEVICE)
# model = MaxVit(2).to(DEVICE)


teacher_model = torch.load('savemodel/model_s_v8.pt', map_location="cpu").to(DEVICE)
optimizer = optim_alg.Ranger(student_model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = CosineDecayWarmup(optimizer=optimizer, 
                              lr=1e-3, 
                              warmup_len=int(epochs*0.1) * len(trainLoader), 
                              total_iters=epochs * len(trainLoader))
criterion = KdCeLoss().to(DEVICE)
criterion_ce = CeLoss().to(DEVICE)


early_stopping = EarlyStopping(patience=30, verbose=False)
scaler = torch.cuda.amp.GradScaler(enabled=True)

if __name__ == "__main__":
    folderCheck([model_path, "eval_fig"])
    best = 0
    train_losses = []
    valid_losses = []

    train_mtx = []
    valid_mtx = []

    for e in range(epochs):

        b_train_loss, b_train_mtx = train(now_ep=e,
                                          teacherNet=teacher_model,
                                          studentNet=student_model,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          dataloader=trainLoader,
                                          criterion=criterion,
                                          scaler=scaler,
                                          DEVICE=DEVICE)
        
        b_valid_loss, b_valid_mtx = evaluate(mode="valid",
                                            model=student_model,
                                            dataloader=vaildLoader,
                                            criterion=criterion_ce,
                                            DEVICE=DEVICE)
        
        train_losses.append(b_train_loss)
        valid_losses.append(b_valid_loss)

        train_mtx.append(b_train_mtx)
        valid_mtx.append(b_valid_mtx)

        early_stopping(b_valid_loss)

        if b_valid_mtx >= best:
            best = b_valid_mtx
            torch.save(student_model, model_path+'/model_s_%s.pt'%model_verison)
            input_x = torch.rand(1, 1, image_size[1], image_size[0]).to(DEVICE)
            traced_model = traced_func(student_model, saved_path=model_path+'/model_s_%s_trace.pt'%model_verison, 
                                       X=input_x)
        
        if early_stopping.early_stop:
            print("Early Stopping !! ")
            break

        gc.collect()
    # end training
    drow(train_losses, valid_losses, 
         train_mtx, valid_mtx,
         name="train_curve")