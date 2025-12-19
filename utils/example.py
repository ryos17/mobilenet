import torch
import torch.nn as nn


class SmallVWWModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_filters = 32
        kernel_size = 3
        stride = 2
        self.conv = nn.LazyConv2d(
            num_filters,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            # Bias must be False until changes to make it i32 from i16
            bias=False,
            padding=0,
        )
        stride = 1
        self.depthwiseconv2d = nn.LazyConv2d(
            num_filters,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            groups=num_filters,
            bias=False,
            padding=0,
        )
        self.pointwiseconv2d = nn.LazyConv2d(
            num_filters * 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
            padding=0,
        )
        self.depthwiseconv2d2 = nn.LazyConv2d(
            num_filters * 2,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            groups=num_filters * 2,
            bias=False,
            padding=0,
        )
        self.pointwiseconv2d2 = nn.LazyConv2d(
            num_filters * 2 * 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
            padding=0,
        )

    def forward(self, x):
        x = NHWC_to_NCHW()(x)
        x = self.conv(x)
        x = nn.ReLU()(x)
        x = self.depthwiseconv2d(x)
        x = nn.ReLU()(x)
        x = self.pointwiseconv2d(x)
        x = nn.ReLU()(x)
        x = self.depthwiseconv2d2(x)
        x = nn.ReLU()(x)
        x = self.pointwiseconv2d2(x)
        x = nn.ReLU()(x)
        x = NCHW_to_NHWC()(x)
        return x

class NHWC_to_NCHW(nn.Module):
    def forward(self, x):
        return torch.permute(x, (0, 3, 1, 2))


class NCHW_to_NHWC(nn.Module):
    def forward(self, x):
        return torch.permute(x, (0, 2, 3, 1))