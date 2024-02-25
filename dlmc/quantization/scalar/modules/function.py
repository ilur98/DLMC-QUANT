from abc import ABCMeta, abstractmethod
from fnmatch import fnmatch
from os import stat
import torch
import torch.nn.functional as F 
from ..ops import get_qparams_tensor, get_qparams_output
from ..utils import emulate_quantize, get_qrange

class FunUniformQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, offset, min_val, max_val):
        weight_q = emulate_quantize(weight, scale, offset, min_val, max_val)
        ctx.save_for_backward(weight, scale)
        ctx.other = min_val, max_val
        return weight_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, scale = ctx.saved_tensors
        g, min_val, max_val = ctx.other
        q_weight = weight / scale
        position_min = (q_weight <= min_val).float()
        position_max = (q_weight >= max_val).float()
        position_middle = torch.ones(weight.shape).to(position_max.device) - position_min - position_max
        grad_weight = position_middle * weight

        return grad_weight, None, None, None, None

class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, offset, min_val, max_val, g):
        weight_q = emulate_quantize(weight.detach(), scale, offset, min_val, max_val)
        ctx.save_for_backward(weight, scale)
        ctx.other = g, min_val, max_val
        return weight_q
    
    @staticmethod
    def backward(ctx, grad_weight):
        weight, scale = ctx.saved_tensors
        g, min_val, max_val = ctx.other
        q_weight = weight / scale
        position_min = (q_weight < min_val).float()
        position_max = (q_weight > max_val).float()
        position_middle = torch.ones(weight.shape, device = position_max.device) - position_min - position_max
        grad_scale = ((min_val * position_min + max_val * position_max + 
                      position_middle * (-q_weight + q_weight.round())) * grad_weight).sum().unsqueeze(dim=0) * g
        grad_weight = position_middle * grad_weight
        
        return grad_weight, grad_scale, None, None, None , None

class FunRootQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, offset, min_val, max_val):
        weight_q = emulate_quantize(weight.detach(), scale, offset, min_val, max_val)
        ctx.save_for_backward(weight, scale)
        
        return weight_q

    @staticmethod
    def backward(ctx, grad_weight):
        
        return grad_weight

class FunLQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, offset, min_val, max_val, g):
        return weight

    @staticmethod
    def backward(ctx, grad_weight):
        return grad_weight, None, None, None, None , None