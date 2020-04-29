import torch
from torch import nn

class reshape(nn.Module):
    '''a torch layer to reshape the input into size = shape = type list'''
  def __init__(self, shape):
      super(reshape, self).__init__()
      self.shape = shape
  def forward(self, x): return x.reshape(self.shape)

class convblock(nn.Module):
    '''
    a convolutional block used in the model:
    conv(in, out) -> batchnorm(out) -> relu
    '''
  def __init__(self, in_:int, out:int):
    super().__init__()
    self.conv1 = nn.Conv2d(in_, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.bn = nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class downsamp(nn.Module):
    ''' a downsampling block. using adaptive max pooling so select the size to be outputted and the scale you would like output ie out (3,10,10) is a downsamp(3, 10).
    '''
  def __init__(self, size:int, scale:int=2):
    super().__init__()
    self.pool = nn.AdaptiveMaxPool2d(scale)
    self.bn = nn.BatchNorm2d(size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.relu = nn.ReLU(inplace = True)

  def forward(self,x):
    x = self.pool(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class Upsample(nn.Module):
    '''
    upsample by scale = scale. Ins and outs are input. Upsampling method is nearest neighbour.
    '''
  def __init__(self, in_:int, out:int, scale:int=2):
    super().__init__()
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    self.bn = nn.BatchNorm2d(in_, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.conv1 = nn.Sequential(
            nn.Conv2d(in_, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
    )

  def forward(self, x):
    x = self.upsample(x)
    x = self.bn(x)
    x = self.conv1(x)
    return x
