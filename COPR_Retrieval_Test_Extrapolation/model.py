import torch
import torch.nn as nn
import torchvision.models as models

class COPR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_features=512
        self.cfg = cfg
        self.net_regdesc = nn.Sequential(*[nn.Linear(in_features+4+3, in_features+4+3),nn.GELU(),
                                           nn.Linear(in_features+4+3, in_features+4+3),nn.GELU(),
                                           nn.Linear(in_features+4+3, in_features+4+3),nn.GELU(),
                                           nn.Linear(in_features+4+3, in_features+4+3),nn.GELU(),
                                           nn.Linear(in_features+4+3, in_features+4+3),nn.GELU(),
                                           nn.Linear(in_features+4+3, in_features+4+3),nn.GELU(),
                                           nn.Linear(in_features+4+3, in_features+4+3),nn.GELU(),
                                           nn.Linear(in_features+4+3, in_features)])
        #self.dropout = nn.Dropout(0.3)

    def _get_backbone(self):
        backbone, concat_layer = None, None
        if self.cfg.model_paramsCOPR.backbone_net == 'resnet34COPR':
            in_features=512 
        return backbone, concat_layer

    def _forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, feat1, RT):
        featcat = torch.cat((feat1, RT), 0)
        feat2=self.net_regdesc(featcat)
        #feat2=self.dropout(self.net_regdesc(featcat))
        
        return feat2

class RelPoseNetOrg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_features=512
        self.backbone, self.concat_layer = self._get_backbone()
        self.net_q_fc = nn.Linear(self.concat_layer.in_features, 4)
        self.net_t_fc = nn.Linear(self.concat_layer.in_features, 3)
        self.dropout = nn.Dropout(0.3)

    def _get_backbone(self):
        backbone, concat_layer = None, None
        if self.cfg.model_paramsrelposenetorg.backbone_net == 'resnet34_originalmodel':
            backbone = models.resnet34(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        return backbone, concat_layer

    def _forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        if (self.cfg.experiment_params.loss_type == 'relativepose'):
            feat1 = self._forward_one(x1)
            return feat1, None, None, None
        
        elif (self.cfg.experiment_params.loss_type == 'triplet' or self.cfg.experiment_params.loss_type == 'distance'): # Since the original relposenet doesn't nomralize but while traning with triplet and distance-based loss I normalize feature vectors 
            feat1 = self._forward_one(x1)
            feat1 = nn.functional.normalize(feat1)
            return feat1, None, None, None
        
    def forward_relpose(self, x1_desc, x2_desc):
        feat = torch.cat((x1_desc, x2_desc), 0)
        q_est = self.net_q_fc(self.dropout(self.concat_layer(feat)))
        t_est = self.net_t_fc(self.dropout(self.concat_layer(feat)))
        return q_est, t_est
