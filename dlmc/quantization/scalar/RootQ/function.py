import torch
import torch.nn as nn
import torch.nn.functional as F

class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        #print(torch.max(x), torch.min(x))
        return x.sgn()
    @staticmethod
    def backward(ctx, g):
        return g 


def clipping(x, upper, lower):
    # clip lower
    x = x + F.relu(lower - x)
    # clip upper
    x = x - F.relu(x - upper)
    return x

def torch_phi_function(x, mi, alpha, delta):
    # alpha = torch.where(alpha >= 0.5, torch.tensor([0.5]).cuda(), alpha)
    # alpha = torch.where(alpha <= 1e-4, torch.tensor([1e-4]).cuda(), alpha)
    alpha = alpha + F.relu(1e-4 - alpha)
    alpha = alpha - F.relu(alpha - 1)
    x = x - mi
    sgn = x / (torch.abs(x) + 1e-5)
    k = 2 / delta
    x =  torch.pow(k * abs(x) + 1e-5, alpha) * sgn

    return x  


class phi_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mi, alpha, delta):
        #alpha = torch.where(alpha >= 0.5, torch.tensor([0.5]).cuda(), alpha)
        #alpha = torch.where(alpha <= 1e-4, torch.tensor([1e-4]).cuda(), alpha)
#
        ctx.save_for_backward(x, mi, alpha, delta)
        x = 2 * (x - mi) / delta
        deltax = torch.max(x) - torch.min(x) + 1e-6
        x = (x/deltax + 0.5)
        return x.round() * 2 - 1

    @staticmethod
    def backward(ctx, g):
        x, mi, alpha, delta = ctx.saved_tensors
        x = 2 * (x - mi) / delta
        sgn = x / (abs(x) + 1e-6)
        grad_x = ((abs(x) + 1e-2) ** (alpha - 1)) * alpha * 2 / delta * g
        grad_alpha = torch.log(abs(x)+1e-2) * ((abs(x)+1e-2) ** alpha) * sgn * g
        grad_delta = -1 * grad_x * x

        return grad_x, None, grad_alpha, grad_delta    

def sgn(x):
    x = RoundWithGradient.apply(x)

    return x

def dequantize(x, lower_bound, delta, interval):

    x =  ((x+1)/2 + interval) * delta + lower_bound

    return x
