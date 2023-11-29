import torch
import torch.nn as nn
from encoder import res50, res18
from vit_model import spatial
from utils import Branch_Attention, Multiscale_Feature_Module, aux_weight_layer

ca=Branch_Attention(channel=1024,ratio=16)
ms=Multiscale_Feature_Module(out_channels=1024)

class aux_net(nn.Module):
    def __init__(self,res=res18,AWL=aux_weight_layer):
        super(aux_net,self).__init__()
        self.res=res
        self.AWL=AWL(channel=14,ratio=7)
        self.down_conv=nn.Conv2d(14,3,1,bias=False)

    def forward(self,x):
        x=self.AWL(x)
        x=self.res(x)

        return x

class rsi_aux_ca_ms(nn.Module):
    def __init__(self, num_classes, ca_module=ca, ms_module=ms, aux_net=aux_net, rsi_features=res50,
                 need_features=False):
        super(rsi_aux_ca_ms, self).__init__()
        self.rsi_features = rsi_features
        self.aux_net = aux_net()
        self.ca_module = ca_module
        self.ms_module = ms_module
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.down_conv0 = nn.Conv2d(2048, 512, 1)
        self.fc_out = nn.Linear(1024, num_classes)
        self.need_features = need_features

    def forward(self, x1, x2):
        rsi_features = self.rsi_features(x1)
        rsi_features = self.down_conv0(rsi_features)
        aux_features = self.aux_net(x2)
        features = torch.cat([rsi_features, aux_features], dim=1)
        weights = self.ca_module(features)
        fusion1 = rsi_features + rsi_features * weights[:, 0, :, :].unsqueeze(1)
        fusion2 = aux_features + aux_features * weights[:, 1, :, :].unsqueeze(1)
        fusion = torch.cat([fusion1, fusion2], dim=1)
        out_features = self.ms_module(fusion)
        out = self.avg(out_features)
        out = out.view(out.size(0), -1)
        out = self.fc_out(out)

        if self.need_features:
            return out_features, out

        return out

class rsi_aux_ca_ms_spatial(nn.Module):
    def __init__(self, features=rsi_aux_ca_ms, spatial=spatial):
        super(rsi_aux_ca_ms_spatial, self).__init__()
        self.features = features(ca_module=ca, ms_module=ms, num_classes=9, need_features=True)
        self.spatial = spatial
        self.project = nn.Conv2d(1024, 768, 1)

    def forward(self, x1, x2):
        out_features, _ = self.features(x1, x2)
        out_features = self.project(out_features).flatten(2).transpose(1, 2)
        out = self.spatial(out_features)

        return out


MMFF=rsi_aux_ca_ms_spatial()