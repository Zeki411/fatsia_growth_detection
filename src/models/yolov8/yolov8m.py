import torch
import torch.nn as nn

from ultralytics.nn.modules import Detect


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
    
    
class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
    
    
class BackBone(nn.Module):
    """Feature extraction backbone for YOLO model."""
    
    def __init__(self):
        super().__init__()

        self.conv1 = Conv(c1=3, c2=48, k=3, s=2)
        self.conv2 = Conv(c1=48, c2=96, k=3, s=2)

        self.c2f1 = C2f(c1=96, c2=96, n=2, shortcut=True)

        self.conv3 = Conv(c1=96, c2=192, k=3, s=2)

        self.c2f2 = C2f(c1=192, c2=192, n=4, shortcut=True)

        self.conv4 = Conv(c1=192, c2=384, k=3, s=2)  # P4 output point

        self.c2f3 = C2f(c1=384, c2=384, n=4, shortcut=True)

        self.conv5 = Conv(c1=384, c2=576, k=3, s=2)

        self.c2f4 = C2f(c1=576, c2=576, n=2, shortcut=True)

        self.sppf = SPPF(c1=576, c2=576, k=5)

    def forward(self, x):
        """Forward pass through the backbone."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c2f1(x)
        p3 = self.conv3(x)
        x = self.c2f2(p3)
        p4 = self.conv4(x)  # Extract P4 feature map
        x = self.c2f3(p4)
        p5 = self.conv5(x)
        x = self.c2f4(p5)
        x = self.sppf(x)

        return x, p3, p4, p5  # Return both final feature map and P3, P4, P5
        

class DetectHead(nn.Module):
    
    def __init__(self, nc: int = 80):
        super().__init__()
        
        self.ups1 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.cc1 = Concat(dimension=1)
        self.c2f1 = C2f(c1=960, c2=384, n=2, shortcut=False)
        
        self.ups2 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.cc2 = Concat(dimension=1)
        self.c2f2 = C2f(c1=384, c2=192, n=2, shortcut=False)
        
        self.conv1 = Conv(c1=192, c2=192, k=3, s=2)
        self.cc3 = Concat(dimension=1)
        self.c2f3 = C2f(c1=576, c2=384, n=2, shortcut=False)
        
        self.conv2 = Conv(c1=384, c2=384, k=3, s=2)
        self.cc4 = Concat(dimension=1)
        self.c2f4 = C2f(c1=384, c2=768, n=2, shortcut=False)
        
        # self.det = Detect(nc=nc)
        
        
    def forward(self, x, p3, p4, p5):
        
        h1 = self.ups1(x)
        h1 = self.cc1([h1, p4])
        h1 = self.c2f1(h1)
        
        h2 = self.ups2(h1)
        h2 = self.cc2([h2, p3])
        h2 = self.c2f2(h2)
        
        h3 = self.conv1(h2)
        h3 = self.cc3([h3, h1])
        h3 = self.c2f3(h3)
        
        h4 = self.conv2(h3)
        h4 = self.cc4([h4, p5])
        h4 = self.c2f4(h4)
        
        
        # y = self.det([h2, h3, h4])
        
        # return y
        
        return h2, h3, h4
    
class Yolom(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = BackBone()
        self.head = DetectHead()

    def forward(self, x):
        """Forward pass through the full model."""
        backbone_out, p3, p4, p5 = self.backbone(x)  # Extract features and P4
        # final_out = self.head(backbone_out, p3, p4, p5)
        h1, h2, h3 = self.head(backbone_out, p3, p4, p5)
        # return final_out
        return h1, h2, h3


if __name__ == "__main__":
    # generate example im data 3x640x640
    
    input = torch.randn(1, 3, 640, 640)
    
    # model = BackBone()
    model = Yolom()
    # print(model)
    
    # print(input.shape)
    
    output = model(input)
    print(output.shape)
        