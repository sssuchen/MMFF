import torch
import torch.nn as nn

class aux_weight_layer(nn.Module):
    def __init__(self,channel,ratio):
        super(aux_weight_layer, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Conv2d(channel,channel*ratio,1,bias=False),
            nn.BatchNorm2d(channel*ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel*ratio,channel,1,bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.down_conv=nn.Conv2d(channel,3,1,bias=False)

    def forward(self,x):
        b,c,_,_=x.size()
        y=self.avg_pool(x)
        y=self.fc(y)
        y=x*y.expand_as(x)
        y=self.down_conv(y)
        return y

class Branch_Attention(nn.Module):
    def __init__(self,channel,ratio=16):
        super(Branch_Attention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        # replace fc with 1x1 Conv
        self.fc1=nn.Conv2d(channel,channel//ratio,1,bias=False)
        self.relu=nn.ReLU()
        self.fc2=nn.Conv2d(channel//ratio,2,1,bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        avg_out=self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out=self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out=avg_out+max_out
        return self.sigmoid(out)



class Multiscale_Feature_Module(nn.Module):
    def __init__(self,out_channels):
        super(Multiscale_Feature_Module, self).__init__()
        self.conv0_1=nn.Conv2d(out_channels,out_channels,(1,3),padding=(0,1),groups=out_channels)
        self.conv0_2=nn.Conv2d(out_channels,out_channels,(3,1),padding=(1,0),groups=out_channels)

        self.conv1_1=nn.Conv2d(out_channels,out_channels,(1,7),padding=(0,3),groups=out_channels)
        self.conv1_2=nn.Conv2d(out_channels,out_channels,(7,1),padding=(3,0),groups=out_channels)

        self.conv2_1=nn.Conv2d(out_channels,out_channels,(1,11),padding=(0,5),groups=out_channels)
        self.conv2_2=nn.Conv2d(out_channels,out_channels,(11,1),padding=(5,0),groups=out_channels)

        self.conv3=nn.Conv2d(out_channels,out_channels,1)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        u=x.clone()
        attn_0=self.conv0_1(x)
        attn_0=self.conv0_2(attn_0)

        attn_1=self.conv1_1(x)
        attn_1=self.conv1_2(attn_1)

        attn_2=self.conv2_1(x)
        attn_2=self.conv2_2(attn_2)

        attn=x+attn_0+attn_1+attn_2
        attn=self.conv3(attn)

        return attn*u