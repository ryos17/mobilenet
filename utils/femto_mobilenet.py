import torch.nn as nn
from torchsummary import summary

class FemtoMobileNetV1(nn.Module):
    def __init__(self, ch_in=3, n_classes=2, kernel_size=3, alpha=1.0):
        super(FemtoMobileNetV1, self).__init__()
        self.alpha = alpha

        # Full convolution block
        def conv_full(inp, oup, stride):
            oup_scaled = int(oup * alpha)
            return nn.Sequential(
                nn.LazyConv2d(inp, oup_scaled, kernel_size, stride, padding=0, bias=False),
                nn.BatchNorm2d(oup_scaled),
                nn.ReLU(inplace=True)
            )

        # Depthwise separable convolution block
        def conv_ds(inp, oup, stride):
            inp_scaled = int(inp * alpha)
            oup_scaled = int(oup * alpha)
            return nn.Sequential(
                # Depthwise convolution
                nn.LazyConv2d(inp_scaled, kernel_size, stride, padding=0, groups=inp_scaled, bias=False),
                nn.BatchNorm2d(inp_scaled),
                nn.ReLU(inplace=True),

                # Pointwise convolution
                nn.LazyConv2d(oup_scaled, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup_scaled),
                nn.ReLU(inplace=True),
            )

        # Model architecture
        self.model = nn.Sequential(
            conv_full(ch_in, 32, 2),
            conv_ds(32, 64, 1),
            conv_ds(64, 128, 2),
            conv_ds(128, 128, 1),
            conv_ds(128, 256, 2),
            conv_ds(256, 256, 1),
            conv_ds(256, 512, 2),
            conv_ds(512, 512, 1),
            conv_ds(512, 512, 1),
            conv_ds(512, 512, 1),
            conv_ds(512, 512, 1),
            conv_ds(512, 512, 1),
            conv_ds(512, 1024, 2),
            conv_ds(1024, 1024, 1),
            nn.AvgPool2d(7)  
        )
        self.fc = nn.Linear(int(1024 * alpha), n_classes)
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, int(1024 * self.alpha))
        x = self.fc(x)
        x = self.softmax(x)
        return x

if __name__=='__main__':
    # Check model
    model = FemtoMobileNetV1(alpha=0.5)
    summary(model, input_size=(3, 224, 224), device='cpu')
    