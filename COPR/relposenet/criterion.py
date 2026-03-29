from abc import ABC
import torch.nn as nn


class RelPoseCriterion(nn.Module, ABC):

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.loss = nn.MSELoss()
        
    def forward(self, feat_gt, feat):
        scale=1000
        pose_loss = self.loss(feat_gt, feat) * scale
        
        return pose_loss