import torch.nn as nn


class FemtoMobileNetV1(nn.Module):
    def __init__(self, ch_in=3, n_classes=2, alpha=1.0):
        super(FemtoMobileNetV1, self).__init__()
        self.alpha = alpha
        self._first_forward = True

        # Full convolution block
        def conv_full(inp, oup, kernel=(3, 3), stride=2, padding=1):
            oup_scaled = int(oup * alpha)
            return nn.Sequential(
                nn.Conv2d(inp, oup_scaled, kernel, stride, padding, bias=True),
                nn.BatchNorm2d(oup_scaled),
                nn.ReLU(inplace=True)
            )

        # Depthwise separable convolution block
        def conv_ds(inp, oup, kernel=(3, 3), stride=1, padding=1):
            inp_scaled = int(inp * alpha)
            oup_scaled = int(oup * alpha)
            return nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(inp_scaled, inp_scaled, kernel, stride, padding, groups=inp_scaled, bias=True),
                nn.BatchNorm2d(inp_scaled),
                nn.ReLU(inplace=True),

                # Pointwise convolution
                nn.Conv2d(inp_scaled, oup_scaled, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(oup_scaled),
                nn.ReLU(inplace=True),
            )

        # Model architecture
        self.model = nn.Sequential(
            # CPU
            conv_full(inp=ch_in, oup=32, stride=2, padding=1),
            conv_ds(inp=32, oup=64, stride=1, padding=1),
            conv_ds(inp=64, oup=128, stride=2, padding=1),
            conv_ds(inp=128, oup=128, stride=1, padding=1),
            conv_ds(inp=128, oup=256, stride=1, padding=1),

            # SPU
            conv_ds(inp=256, oup=256, stride=1, padding=0),
            conv_ds(inp=256, oup=512, stride=1, padding=0),
            conv_ds(inp=512, oup=512, stride=1, padding=0),
            conv_ds(inp=512, oup=512, stride=1, padding=0),
            conv_ds(inp=512, oup=512, stride=1, padding=0),
            conv_ds(inp=512, oup=512, stride=1, padding=0),
            conv_ds(inp=512, oup=512, stride=1, padding=0),
            conv_ds(inp=512, oup=1024, stride=1, padding=0),
            conv_ds(inp=1024, oup=1024, stride=1, padding=0),

            # CPU
            nn.AdaptiveAvgPool2d(1)  
        )
        self.fc = nn.Linear(int(1024 * alpha), n_classes)
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, x):
        if self._first_forward:
            print(f"\nInput shape: {x.shape}")
            
            # Process through each layer and print sizes
            for i, layer in enumerate(self.model):
                x = layer(x)
                print(f"After layer {i} ({type(layer).__name__}): {x.shape}")
            
            x = x.view(-1, int(1024 * self.alpha))
            print(f"After view: {x.shape}")
            x = self.fc(x)
            print(f"After FC: {x.shape}")
            x = self.softmax(x)
            print(f"After softmax: {x.shape}")
            self._first_forward = False
        else:
            x = self.model(x)
            x = x.view(-1, int(1024 * self.alpha))
            x = self.fc(x)
            x = self.softmax(x)

        return x
