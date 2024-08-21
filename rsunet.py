

import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvMod(nn.Module):
    """
    Convolution module.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvMod, self).__init__()
        
        # Convolutions.
        self.conv1 = nn.Conv3d(in_channels,  out_channels, kernel_size = (1,3,3), stride = 1, padding = (0,1,1))
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size = (3,3,3), stride = 1, padding = 1)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size = (3,3,3), stride = 1, padding = 1)
        # BatchNorm.
        self.bn1 = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.01)
        self.bn2 = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.01)
        self.bn3 = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.01)
        
        # Activation function.
        self.activation = nn.ELU()
    
    def forward(self, x):
        # Conv 1.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        skip = x
        # Conv 2.
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        # Conv 3.
        x = self.conv3(x)
        x = x + skip
        x = self.bn3(x)
        return self.activation(x)

    
class RSUNet(nn.Module):
    def __init__(self, in_ch = 1, out_ch = 1):
        super(RSUNet, self).__init__()
        self.conv0 = nn.Conv3d(in_ch, 28, kernel_size = (1,5,5),padding = (0,2,2))
        self.elu0 = nn.ELU()
        self.conv1 = ConvMod(28, 36)
        self.pool1 = nn.MaxPool3d(kernel_size = (1,2,2))
        self.conv2 = ConvMod(36, 48)
        self.pool2 = nn.MaxPool3d(kernel_size = (1,2,2))
        self.conv3 = ConvMod(48, 64)
        self.pool3 = nn.MaxPool3d(kernel_size = (1,2,2))
        self.conv4 = ConvMod(64, 80)
        self.pool4 = nn.MaxPool3d(kernel_size = (1,2,2))

        # deconvolution
        self.up5 = nn.ConvTranspose3d(80,64,(1,2,2),stride = (1,2,2), padding = 0)
        self.conv5 = ConvMod(64, 64)
        self.up6 = nn.ConvTranspose3d(64,48,(1,2,2),stride = (1,2,2), padding = 0)
        self.conv6 = ConvMod(48, 48)
        self.up7 = nn.ConvTranspose3d(48,36,(1,2,2),stride = (1,2,2), padding = 0)
        self.conv7 = ConvMod(36, 36)
        self.up8 = nn.ConvTranspose3d(36,28,(1,1,1),stride = (1,1,1), padding = 0)
        self.conv8 = ConvMod(28, 28)
        self.conv9 = nn.Conv3d(28, out_ch, (1,5,5), padding=(0,2,2))
        self.sig0 = nn.Sigmoid()
        
        
    def forward(self,x):
        c0 = self.conv0(x)
        c0 = self.elu0(c0)
        
        c1 = self.conv1(c0)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)   #c4 size is 1,80,18,20,20
        p4 = c4 # no pooling needed
    
        up_5 = self.up5(p4)
        c5 = self.conv5(up_5 + c3)
        
        up_6 = self.up6(c5)
        c6 = self.conv6(up_6 + c2)
        
        up_7 = self.up7(c6)
        c7 = self.conv7(up_7 + c1)
        
        up_8 = self.up8(c7) #no upsample here, just channel collapse
        c8 = self.conv8(up_8 + c0)
        c9 = self.conv9(c8)
        
#         out = self.sig0(c9)
        out = c9 #this is for in case we want to introduce loss weights via BCEwithlogitloss
  
        return(out)

