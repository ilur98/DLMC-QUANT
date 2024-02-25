import torch.nn as nn
import torch
import torch.nn.functional as F


class conv_bn_act(nn.Module):
    def __init__(self, inchannels, outchannels, kernelsize, stride=1, dilation=1, groups=1, bias=False, bn_momentum=0.99):
        super().__init__()
        self.block = nn.Sequential(
            SameConv(inchannels, outchannels, kernelsize, stride, dilation, groups, bias=bias),
            nn.BatchNorm2d(outchannels, momentum=1-bn_momentum),
            swish()
        )

    def forward(self, x):
        return self.block(x)


class SameConv(nn.Module):
    def __init__(self, inchannels, outchannels, kernelsize, stride=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.ka = kernelsize // 2
        self.kb = self.ka - 1 if kernelsize % 2 == 0 else self.ka
        #self.pad  = nn.ConstantPad2d([ka, kb, ka, kb], 0)
        #print(stride)
        self.conv = nn.Conv2d(inchannels, outchannels, kernelsize, stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = F.pad(x, [self.ka, self.kb, self.ka, self.kb])
        #out = self.conv(x)
        return self.conv(x)
    #def count_your_model(self, x, y):
     #   return y.size(2) * y.size(3) * y.size(1) * self.weight.size(2) * self.weight.size(3) / self.groups


class swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):
    def __init__(self, inchannels, mid):
        super().__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(inchannels, mid),
            swish(),
            nn.Linear(mid, inchannels)
        )

    def forward(self, x):
        out = self.AvgPool(x)
        out = out.view(x.size(0), -1)
        out = self.SEblock(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return x * torch.sigmoid(out)


class drop_connect(nn.Module):
    def __init__(self, survival=0.8):
        super().__init__()
        self.survival = survival

    def forward(self, x):
        if not self.training:
            return x

        random = torch.rand((x.size(0), 1, 1, 1), device=x.device) # 涉及到x的属性的步骤，直接挪到forward
        random += self.survival
        random.requires_grad = False
        return x / self.survival * torch.floor(random)