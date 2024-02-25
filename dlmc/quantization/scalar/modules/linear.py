from torch.nn import Linear
import torch.nn.functional as F

from .base import QBase


class QLinear(QBase, Linear):
    def __init__(self, *args, qconfig=None, **kwargs):
        Linear.__init__(self, *args, **kwargs)
        QBase.__init__(self, qconfig)

    def _forward_func(self, input, weight):
        return F.linear(input, weight, self.bias)
