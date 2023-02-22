import torch
import torch.nn as nn
import torchvision.ops as ops
#from .weight_init import constant_init, kaiming_init
#from .norm import build_norm_layer

class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1):
        super(ConvModule, self).__init__()

        # self.conv = nn.Conv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     groups=groups,
        #     bias=False)


        self.conv = ops.DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False)

        self.offset = nn.Conv2d(
            in_channels,
            2*kernel_size*kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True)

        #self.norm = nn.BatchNorm2d(out_channels)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out', nonlinearity='relu')
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

        nn.init.constant_(self.offset.weight, 0)
        if hasattr(self.offset, 'bias'):
            nn.init.constant_(self.offset.bias, 0)

    def forward(self, x):

        offset = self.offset(x)
        x = self.conv(x, offset)
        x = self.norm(x)
        x = self.relu(x)

        return x
