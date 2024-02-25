from torch.nn import Linear
import torch.nn.functional as F

from .base import FSPTQBase


class FSPTQLinear(FSPTQBase, Linear):
    def __init__(self, *args, qconfig=None, **kwargs):
        Linear.__init__(self, *args, **kwargs)
        FSPTQBase.__init__(self, qconfig)

    def _forward_func(self, input, weight):
        out = F.linear(input, weight, self.bias)
        #print(out)
        return out
