import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def activation_func(activation):
    return nn.ModuleDict({
        'selu': nn.SELU(inplace=True),
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'sigmoid': nn.Sigmoid(),
        'prelu': nn.PReLU(),
        'softmax': nn.Softmax(dim=1),
        'gelu': nn.GELU()})[activation]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channel, k_size, activation='relu', s=1, pad=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channel,
                              k_size, padding=pad, stride=s, dilation=dilation)
        self.batchNorm = nn.BatchNorm2d(out_channel)
        self.actfunction = activation_func(activation)
        self.act_name = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.actfunction(x)
        x = self.batchNorm(x)
        return x

class DWCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride, bias):
        super(DWCNNBlock, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_ch,
                                    bias=bias)
        
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=bias)
        
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self,input):
        out = self.relu6(self.depth_conv(input))
        out = self.point_conv(out)
        return out

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.K = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.V = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.up_channel = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=1)

    def _softmax(self, x):
        with torch.no_grad():
            max_val = torch.max(x, dim=-1, keepdim=True)[0]
            torch.sub(x, max_val, out=x)
            torch.exp(x, out=x)
            summed = torch.sum(x, dim=-1, keepdim=True)
            x /=summed
            return x
    
    def forward(self, input):
        b, c, w, h= input.size() # (1, 64, 3, 3)

        q = self.Q(input).view(b, -1, c) # (1, 9, 32)
        k = self.K(input).view(b, -1, c) # (1, 9, 32)
        v = self.V(input).view(b, -1, c) # (1, 9, 32)

        kt = torch.transpose(k, 2, 1) # (1, 32, 9)
        score_map = torch.bmm(q, kt)  # (1, 9, 9)
        score_map = self._softmax(score_map)

        score_map = torch.bmm(score_map, v).view(b, c, w, h)# (1, 32, 3, 3)
        score_map = self.up_channel(score_map)# (1, 64, 3, 3)
        
        return input + score_map # (1, 64, 3, 3)
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, activation='relu'):
        super().__init__()
        self.convR1 = CNNBlock(in_channels, int(in_channels/4), k_size=1, activation=activation, pad=0)
        self.convR2 = CNNBlock(int(in_channels/4), int(in_channels/4), k_size=3, activation=activation, pad=1)
        self.convR3 = CNNBlock(int(in_channels/4), in_channels, k_size=1, activation=activation, pad=0)
        self.actfunctionR = activation_func(activation)

    def forward(self, x):
        x1 = self.convR1(x)
        x2 = self.convR2(x1)
        x3 = self.convR3(x2)
        res = x + x3
        res = self.actfunctionR(res)
        return res
    
class Se(nn.Module):
    def __init__(self, in_channels, reduce=16):
        super().__init__()
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.rb = ResBlock(in_channels)
        self.se = nn.Sequential(nn.Linear(in_channels, in_channels // reduce, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_channels // reduce, in_channels, bias=False),
                                nn.Sigmoid() )
    def forward(self, input):
        X = input
        X = self.rb(X)
        b, c, _, _= X.size()
        y = self.gp(X).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        y = X * y.expand_as(X)
        out = y + input
        return out

class Resnet(nn.Module):
    def __init__(self, output, weights="IMAGENET1K_V1"):
        super().__init__()
        
        self.stmm = nn.Conv2d(1, 3, kernel_size=1, padding=0, bias=False)
        backbone = models.resnet18(weights=weights)
        # print(backbone)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.pool = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.medium = nn.Conv2d(256, 40, kernel_size=1, padding=0, stride=1, bias=False)
        self.medium_back = nn.Conv2d(40, 256, kernel_size=1, padding=0, stride=1, bias=False)

        self.layer4 = backbone.layer4

        self.deep = nn.Conv2d(512, 576, kernel_size=1, padding=0, stride=1, bias=False)
        self.deep_back = nn.Conv2d(576, 512, kernel_size=1, padding=0, stride=1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, output)
        )

    def forward(self, x):
        x = self.stmm(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.medium(self.layer3(x))
        l4 = self.deep(self.layer4(self.medium_back(l3)))
        x = self.avgpool(self.deep_back(l4))
        x = self.classifier(x)
        return x, l3, l4

class Efficientnet_s(nn.Module):
    def __init__(self, output, weights="IMAGENET1K_V1"):
        super().__init__()
        
        model = models.efficientnet_b0(weights=weights)

        features = model.features
        self.stmm = nn.Conv2d(1, 3, kernel_size=1, padding=0, bias=False)

        self.backbone_A = []
        self.backbone_B = []

        for i, layer in enumerate(features):
            if i == 5:
                break
            self.backbone_A.append(layer)
  
        for i, layer in enumerate(features):
            if i == len(features)-1:
                break
            if i > 4:
                self.backbone_B.append(layer)

        self.backbone_A = nn.Sequential(*self.backbone_A)
        self.medium = nn.Conv2d(80, 40, kernel_size=1, stride=1, padding=0, bias=False)

        self.back_res = nn.Conv2d(40, 80, kernel_size=1, stride=1, padding=0, bias=False)

        self.backbone_B = nn.Sequential(*self.backbone_B)

        self.neck_f = nn.Conv2d(320, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.neck_bn = nn.BatchNorm2d(576)
        self.neck_act = nn.SiLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.3, inplace=True),
            nn.Linear(576, output)
        )

    def forward(self, x):
        x = self.stmm(x)
        x = self.backbone_A(x)
        medium = self.medium(x)
        x = self.backbone_B(self.back_res(medium))

        neck_f = self.neck_f(x)
        neck_bn = self.neck_bn(neck_f)
        neck_act = self.neck_act(neck_bn)

        x = self.avgpool(neck_act)
        x = self.classifier(x)
        return x, medium, neck_f
    
class Efficientnet(nn.Module):
    def __init__(self, output, weights="IMAGENET1K_V1"):
        super().__init__()
        
        model = models.efficientnet_v2_s(weights=weights)
        print(model)
        features = model.features
        self.stmm = nn.Conv2d(1, 3, kernel_size=1, padding=0, bias=False)

        self.backbone_A = []
        self.backbone_B = []

        for i, layer in enumerate(features):
            if i == 5:
                break
            self.backbone_A.append(layer)
  
        for i, layer in enumerate(features):
            if i == len(features)-1:
                break
            if i > 4:
                self.backbone_B.append(layer)

        self.backbone_A = nn.Sequential(*self.backbone_A)
        self.medium = nn.Conv2d(128, 40, kernel_size=(1,1), stride=(1,1), bias=False)
        self.back_res = nn.Conv2d(40, 128, kernel_size=(1,1), stride=(1,1), bias=False)

        self.backbone_B = nn.Sequential(*self.backbone_B)

        # print(self.backbone)
        self.neck_f = nn.Conv2d(256, 576, kernel_size=(1,1), stride=(1,1), bias=False)
        self.neck_bn = nn.BatchNorm2d(576)
        self.neck_act = nn.SiLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(576, output)
        )

    def forward(self, x):
        x = self.stmm(x)
        x = self.backbone_A(x)
       
        medium = self.medium(x)
        # print(medium.size())
        # medium_back = self.back_res_A(medium)
        # x = self.backbone_B(self.back_res_B(medium_back+x))
        x = self.backbone_B(self.back_res(medium))

        neck_f = self.neck_f(x)
        neck_bn = self.neck_bn(neck_f)
        neck_act = self.neck_act(neck_bn)

        x = self.avgpool(neck_act)
        x = self.classifier(x)
        return x, medium, neck_f

class Densenet(nn.Module):
    def __init__(self, output, weights="IMAGENET1K_V1"):
        super().__init__()

        self.model_ft = models.densenet121(weights=weights)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=1, padding=1, bias=False)
        self.model_ft.classifier = nn.Sequential(nn.Linear(1024, output))
    def forward(self, x):
        x = self.conv1(x)
        x = self.model_ft(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, output, weights="IMAGENET1K_V1"):
        super().__init__()

        model = models.mobilenet_v3_small(weights=weights)
 
        features = model.features
        self.stmm = nn.Conv2d(1, 3, kernel_size=1, padding=0, bias=False)

        self.backbone_A = []
        self.backbone_B = []

        for i, layer in enumerate(features):
            if i > 6:
                break
            self.backbone_A.append(layer)

        for i, layer in enumerate(features):
            if i == len(features)-1:
                break
            if i > 6:
                self.backbone_B.append(layer)

        self.backbone_A = nn.Sequential(*self.backbone_A)
        self.backbone_B = nn.Sequential(*self.backbone_B)

        self.neck_f = nn.Conv2d(96, 576, kernel_size=(1,1), stride=(1,1), bias=False)
        self.neck_bn = nn.BatchNorm2d(576)
        self.neck_act = nn.Hardswish(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1024, output)

        )
        
    def forward(self, x):
        x = self.stmm(x)
        medium = self.backbone_A(x)
        x = self.backbone_B(medium)

        neck_f = self.neck_f(x)
        neck_bn = self.neck_bn(neck_f)
        neck_act = self.neck_act(neck_bn)

        x = self.avgpool(neck_act)
        x = self.classifier(x)
        return x, medium, neck_f

class MobileNetL(nn.Module):
    def __init__(self, output, weights="IMAGENET1K_V1"):
        super().__init__()

        model = models.mobilenet_v3_large(weights=weights)
        # print(model)
        features = model.features
        self.stmm = nn.Conv2d(1, 3, kernel_size=1, padding=0, bias=False)

        self.backbone_A = []
        self.backbone_B = []

        for i, layer in enumerate(features):
            if i > 7:
                break
            self.backbone_A.append(layer)
        self.backbone_A.append(nn.Conv2d(80, 40, kernel_size=1, padding=0, bias=False))

        self.backbone_B.append(nn.Conv2d(40, 80, kernel_size=1, padding=0, bias=False))

        for i, layer in enumerate(features):
            if i == len(features)-1:
                break
            if i > 7:
                self.backbone_B.append(layer)

        self.backbone_A = nn.Sequential(*self.backbone_A)
        self.backbone_B = nn.Sequential(*self.backbone_B)

        self.neck_f = nn.Conv2d(160, 576, kernel_size=(1,1), stride=(1,1), bias=False)
        self.neck_bn = nn.BatchNorm2d(576)
        self.neck_act = nn.Hardswish(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, output)

        )
        
    def forward(self, x):
        x = self.stmm(x)
        medium = self.backbone_A(x)
        # print(medium.size())
        x = self.backbone_B(medium)

        neck_f = self.neck_f(x)
        neck_bn = self.neck_bn(neck_f)
        neck_act = self.neck_act(neck_bn)

        x = self.avgpool(neck_act)
        x = self.classifier(x)
        return x, medium, neck_f
    
class MaxVit(nn.Module):
    def __init__(self, output, weights="IMAGENET1K_V1"):
        super().__init__()

        self.model_ft = models.maxvit_t(weights=weights)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=1, padding=0, bias=False)
        self.model_ft.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                 nn.Flatten(start_dim=1, end_dim=-1),
                                                 nn.LayerNorm(512),
                                                 nn.Linear(512, 512),
                                                 nn.Tanh(),
                                                 nn.Linear(512, output, bias=False))
        
    def forward(self, x):
        x = self.conv1(x)
        # print(self.model_ft.blocks[0](self.model_ft.stem(x)).size())
        x = self.model_ft(x)
        return x
if __name__ == "__main__":

    from torchinfo import summary
    DEVICE = torch.device("cpu")

    # import time
    input = torch.randn(1, 1, 380, 256).to(DEVICE)
    model = Efficientnet(2).to(DEVICE)
    
    # print(model)
    summary(model, input_data=input)
    
    output1, output2, output3 = model(input)
    print(output1.size())
    print(output2.size())
    print(output3.size())

