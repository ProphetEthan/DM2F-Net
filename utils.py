import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def better_upsampling(x, y):
    """
    Perform better upsampling operation.

    Args:
    - x (torch.Tensor): Input tensor of shape (N, C, H, W).
    - y (torch.Tensor): Target tensor of shape (N, C, H', W').

    Returns:
    - x_upsampled (torch.Tensor): Upsampled tensor of shape (N, out_ch, H', W').
    """
    # Perform interpolation to match the size of y
    x_upsampled = F.interpolate(x, size=y.size()[2:], mode='nearest', align_corners=None)
    
    # Pad the upsampled tensor
    padding = (3 // 2, int(3 / 2), 3 // 2, int(3 / 2))
    x_upsampled = F.pad(x_upsampled, padding)
    
    # Apply convolution
    conv = nn.Conv2d(x.size()[1], x.size()[1], kernel_size=3, padding=0).to(x.device)
    x_upsampled = conv(x_upsampled)
    
    return x_upsampled

class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate,):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out