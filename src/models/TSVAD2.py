import torch.nn as nn
# from torchlibrosa.stft import Spectrogram, LogmelFilterBank
# from torchlibrosa.augmentation import SpecAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init
# from pytorch_utils import do_mixup, interpolate, pad_framewise_output
import pdb
__all__ = ['TSVAD2']

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if x.shape[2]<2:
            return x
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Conv_Block(nn.Module):
    def __init__(self,in_channels,out_channels,pooling_size) -> None:
        super().__init__()
        self.conv_gate = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.convs = self.conv_block(out_channels,out_channels,pooling_size)
        self.pooling = nn.MaxPool2d(pooling_size)

    def conv_block(self,in_channels,out_channels,pooling_size):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=(1,3-pooling_size),padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self,x,pool=True):
        gate = self.conv_gate(x)
        x = x*gate
        out = self.convs(x)
        if pool:
            out = self.pooling(out)
        return out

def conv_block(in_channels,out_channels,pooling_size):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=(1,3-pooling_size),padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pooling_size)
        )

class TSVAD2(nn.Module):
    def __init__(self,PANN_layers,num_classes,IF_pooling=False):
        super(TSVAD2,self).__init__()
        # self.encoder = nn.Sequential(
        #     conv_block(1,128,pooling_size=2),
        #     conv_block(128,128,pooling_size=2),
        #     conv_block(128,128,pooling_size=1),
        #     conv_block(128,128,pooling_size=1)
        # )
        if IF_pooling:
            in_channel = 64*(2**(PANN_layers-1))* (128//2**(PANN_layers))
        else:
            in_channel = 64*2**(PANN_layers-1) * 128
        self.fc_ = nn.Linear(in_channel,num_classes)

    def forward(self,x,step=1):
        b,c,seq_len,_ = x.shape
        x = torch.permute(x,[0,2,1,3]).reshape(b*seq_len,-1)
        out = self.fc_(x)
        return out

# model = TSVAD1(num_classes=10)
# data =torch.randn(64,862,128)
# out = model(data,step=2)