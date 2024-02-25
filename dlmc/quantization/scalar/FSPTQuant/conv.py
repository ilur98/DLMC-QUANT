from torch.nn import Conv2d
import torch.nn.functional as F

from ...ntuple import _pair
from .base import FSPTQBase


class FSPTQConv2d(FSPTQBase, Conv2d):
    def __init__(self, *args, qconfig=None, **kwargs):
        Conv2d.__init__(self, *args, **kwargs)
        FSPTQBase.__init__(self, qconfig)

    def _forward_func(self, input, weight):
        if self.padding_mode != 'zeros':
            out = F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
            #print(out.mean())
            return out
        #print(input.device, weight.device)
        out = F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        #print(out.mean())
        return out


