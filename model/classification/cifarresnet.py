'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['cifar_resnet20',  'cifar_resnet32',   'cifar_resnet44', 'cifar_resnet56',
           'cifar_resnet110', 'cifar_resnet1202', 'cifar_resnet']
__ref__ = 'https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py'

base_url = 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/'
model_urls = {
    'cifar_resnet20': base_url + 'resnet20-12fca82f.th',
    'cifar_resnet32': base_url + 'resnet32-d509ac18.th',
    'cifar_resnet44': base_url + 'resnet44-014dd654.th',
    'cifar_resnet56': base_url + 'resnet56-4bfd9763.th',
    'cifar_resnet110': base_url + 'resnet110-1d1ed7c2.th',
    'cifar_resnet1202': base_url + 'resnet1202-f3b1deed.th',
}


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def _get_shortcut(in_planes, out_planes, stride, option):

    assert option in ('A', 'B', 'C', 'D')
    # assert in_planes == out_planes or in_planes * 2 == out_planes

    shortcut = nn.Sequential()
    if option == 'D':
        if stride != 1:
            shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )
    elif stride != 1 or in_planes != out_planes:
        if option == 'A':
            """
            For CIFAR10 ResNet paper uses option A.
            """
            pad_planes = (out_planes - in_planes) // 2
            shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::stride, ::stride],
                                                   (0, 0, 0, 0, pad_planes, pad_planes), "constant", 0))
        elif option == 'B':
            shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        elif option == 'C':
            shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    return shortcut


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = _get_shortcut(in_planes, self.expansion * planes, stride, option)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = _get_shortcut(in_planes, self.expansion * planes, stride, option)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, width=16, num_classes=10, option='A'):
        super(CifarResNet, self).__init__()
        self.in_planes = width

        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(block, width, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(block, width*2, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(block, width*4, num_blocks[2], stride=2, option=option)
        self.linear = nn.Linear(self.in_planes, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _cifar_resnet(arch, block, num_blocks, width, num_classes, pretrained, option):
    """Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L223"""
    assert not (pretrained and (num_classes != 10 or option != 'A'))
    model = CifarResNet(block, num_blocks, width, num_classes, option)
    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        checkpoint = load_state_dict_from_url(model_urls[arch])
        state_dict = {key.replace("module.", ''): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    return model


def cifar_resnet20(num_classes=10, pretrained=False, option='A'):
    return _cifar_resnet('cifar_resnet20', BasicBlock, [3, 3, 3], 16, num_classes, pretrained, option)


def cifar_resnet32(num_classes=10, pretrained=False, option='A'):
    return _cifar_resnet('cifar_resnet32', BasicBlock, [5, 5, 5], 16, num_classes, pretrained, option)


def cifar_resnet44(num_classes=10, pretrained=False, option='A'):
    return _cifar_resnet('cifar_resnet44', BasicBlock, [7, 7, 7], 16, num_classes, pretrained, option)


def cifar_resnet56(num_classes=10, pretrained=False, option='A'):
    return _cifar_resnet('cifar_resnet56', BasicBlock, [9, 9, 9], 16, num_classes, pretrained, option)


def cifar_resnet110(num_classes=10, pretrained=False, option='A'):
    return _cifar_resnet('cifar_resnet110', BasicBlock, [18, 18, 18], 16, num_classes, pretrained, option)


def cifar_resnet1202(num_classes=10, pretrained=False, option='A'):
    return _cifar_resnet('cifar_resnet1202', BasicBlock, [200, 200, 200], 16, num_classes, pretrained, option)


def cifar_resnet(num_blocks, width=16, block="BasicBlock", num_classes=10, pretrained=False, option='A'):
    msg = "ResNet block must be `BasicBlock` or `Bottleneck`, got {}".format(block)
    assert block in ("BasicBlock", "Bottleneck"), msg
    assert pretrained is False
    block = BasicBlock if block == "BasicBlock" else Bottleneck

    return _cifar_resnet('cifar_resnet', block, num_blocks, width, num_classes, pretrained, option)
