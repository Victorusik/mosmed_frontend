import torch.nn as nn
import timm

class MultiHeadCTModel(nn.Module):
    def __init__(self, backbone_name="resnet50", num_multiclass=4):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.head_binary = nn.Linear(in_features, 2)
        self.head_multi  = nn.Linear(in_features, num_multiclass)

    def forward(self, x):
        feats = self.backbone(x)
        out_bin  = self.head_binary(feats)
        out_multi = self.head_multi(feats)
        return out_bin, out_multi
