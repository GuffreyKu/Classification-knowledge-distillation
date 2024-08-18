import torch
import torch.nn as nn
import torch.nn.functional as F


class CeLoss(nn.Module):
    def __init__(self):
        super(CeLoss, self).__init__()
        self.alpha = 1
        self.gamma = 2
    def forward(self, pred, label, weight):
        ce = F.cross_entropy(pred, label, weight=weight, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce).mean()
        return focal_loss
    
class KdCeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CeLoss()
        self.T = 2
        self.soft_target_loss_weight = 0.25
        self.hidden_rep_loss_weight = 0.25
        self.ce_loss_weight = 0.5
        # self.cosine_loss = nn.CosineEmbeddingLoss()
        self.KLD_loss = nn.KLDivLoss()

    def forward(self, teacher_pred, student_pred, teacher_embed, student_embed, label, weight):
        #Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_pred / self.T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_pred / self.T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.T**2)
        focal_loss = self.ce_loss(student_pred, label, weight)

        embed_loss_medium = self.KLD_loss(F.log_softmax(torch.flatten(student_embed["medium"], start_dim=1)), 
                                   F.softmax(torch.flatten(teacher_embed["medium"], start_dim=1)))
        
        embed_loss_deep = self.KLD_loss(F.log_softmax(torch.flatten(student_embed["deep"], start_dim=1)), 
                                   F.softmax(torch.flatten(teacher_embed["deep"], start_dim=1)))
        # print("soft_targets_loss: ", soft_targets_loss)
        # print("focal_loss: ", focal_loss)
        # print("embed_loss: ", embed_loss)

        loss = self.soft_target_loss_weight*soft_targets_loss + self.ce_loss_weight*focal_loss + self.hidden_rep_loss_weight*(embed_loss_medium+embed_loss_deep)
        return  loss