import layers
import torch
from torch import nn
from fastai import *
from fastai.vision import *

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.convblock1 = layers.convblock(3,12)
        self.downsamp1 = layers.downsamp(12, 16)
        self.convblock2 = layers.convblock(12,24)
        self.downsamp2 = layers.downsamp(24, 8)
        self.bottleneck = nn.Sequential(nn.Flatten(),
                                        nn.Linear(24 * 8 * 8, 1000)
                                        )
    def forward(self, x):
        x = self.convblock1(x)
        x = self.downsamp1(x)
        x = self.convblock2(x)
        x = self.downsamp2(x)
        x = self.bottleneck(x)
        return x

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(1000, 24 * 8 * 8),
                                        layers.reshape([-1,24,8,8])
                                        )
        self.up1 = layers.PixelShuffle_ICNR(ni=24,nf=12,scale=2,blur=True,norm_type=NormType.Weight,leaky=None)
        self.bn1 = nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.up2 = layers.PixelShuffle_ICNR(ni=12,nf=3,scale=2,blur=True,norm_type=NormType.Weight,leaky=None)
        self.bn2 = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self,x):
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.bn1(x)
        x = self.up2(x)
        x = self.bn2(x)
        return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()

    def encode(self, x): return self.encoder(x)
    def decode(self, x): return torch.clamp(self.decoder(x), min = 0, max = 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.clamp(x, min = 0, max = 1)
