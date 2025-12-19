import torch
import torch.nn as nn
from torchsummary import summary


class FemtoMobileNetV1(nn.Module):
    def __init__(self, ch_in=3, n_classes=2, alpha=1.0):
        super(FemtoMobileNetV1, self).__init__()
        self.alpha = alpha

        # Full convolution block
        def conv_full(inp, oup, kernel, stride, padding):
            oup_scaled = int(oup * alpha)
            return nn.Sequential(
                nn.LazyConv2d(oup_scaled, kernel, stride, padding, bias=False),
                nn.ReLU(inplace=True)
            )

        # Depthwise separable convolution block
        def conv_ds(inp, oup, kernel, stride, padding):
            inp_scaled = int(inp * alpha)
            oup_scaled = int(oup * alpha)
            return nn.Sequential(
                # Depthwise convolution
                nn.LazyConv2d(inp_scaled, kernel, stride, padding, groups=inp_scaled, bias=False),
                nn.ReLU(inplace=True),

                # Pointwise convolution
                nn.LazyConv2d(oup_scaled, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )

        # Model architecture
        self.model = nn.Sequential(
            # CPU
            conv_full(inp=ch_in, oup=32, kernel=(3, 3), stride=2, padding=1),
            conv_ds(inp=32, oup=64, kernel=(3, 3), stride=1, padding=1),
            conv_ds(inp=64, oup=128, kernel=(3, 3), stride=2, padding=1),

            # SPU
            conv_ds(inp=128, oup=128, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=128, oup=256, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=256, oup=256, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=256, oup=512, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=512, oup=512, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=512, oup=512, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=512, oup=512, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=512, oup=512, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=512, oup=512, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=512, oup=1024, kernel=(1, 3), stride=1, padding=0),
            conv_ds(inp=1024, oup=1024, kernel=(1, 3), stride=1, padding=0),

            # CPU
            nn.AdaptiveAvgPool2d(1)  
        )
        self.fc = nn.Linear(int(1024 * alpha), n_classes)
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, x):
        x = NHWC_to_NCHW()(x)
        x = self.model(x)
        x = NCHW_to_NHWC()(x)

        x = x.view(-1, int(1024 * self.alpha))
        x = self.fc(x)
        x = self.softmax(x)

        return x

class NHWC_to_NCHW(nn.Module):
    def forward(self, x):
        return torch.permute(x, (0, 3, 1, 2))

class NCHW_to_NHWC(nn.Module):
    def forward(self, x):
        return torch.permute(x, (0, 2, 3, 1))
