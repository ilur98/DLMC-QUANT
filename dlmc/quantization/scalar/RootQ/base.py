from abc import ABCMeta, abstractmethod
from fnmatch import fnmatch
import torch
import math
from torch.nn import Module
from pathlib import PosixPath
from ..ops import get_qparams_tensor, get_qparams_output
from ..utils import emulate_quantize, get_qrange, round_pass, floor_pass
from .function import *

class RootQBase(Module):
    __metaclass__ = ABCMeta

    qconfig: dict
    wt_min_val: int
    wt_max_val: int
    wt_upper: torch.nn.Parameter
    wt_lower: torch.nn.Parameter
    wt_offset: torch.Tensor
    wt_run_upper: torch.Tensor
    wt_run_lower: torch.Tensor
    wt_alpha: torch.nn.Parameter
    in_min_val: int
    in_max_val: int
    in_upper: torch.nn.Parameter
    in_lower: torch.nn.parameter
    in_run_upper: torch.Tensor
    in_run_lower: torch.Tensor
    in_run_scale: torch.Tensor
    in_offset: torch.Tensor
    in_alpha: torch.nn.Parameter

    def __init__(self, qconfig: dict = None):
        super(RootQBase, self).__init__()
        self.initialize(qconfig)

    def initialize(self, qconfig):
        self.qconfig = qconfig

        self.wt_min_val, self.wt_max_val = get_qrange(
            qconfig['weight']['args']['signed'],
            qconfig['weight']['args']['n_bits']
        )
        self.in_min_val, self.in_max_val = get_qrange(
            qconfig['input']['args']['signed'],
            qconfig['input']['args']['n_bits']
        )
        #self.register_parameter('in_upper', torch.nn.Parameter(torch.tensor(2 ** 3 - 1).float()))
        #self.register_parameter('in_lower', torch.nn.Parameter(torch.tensor((-1) * (2 ** 2)).float()))
        self.register_parameter('in_scale', torch.nn.Parameter(torch.tensor(1.).float()))
        #self.register_parameter('in_alpha', torch.nn.Parameter(torch.tensor(1./7).float()))
        self.register_buffer('in_offset', None)
        self.register_buffer('in_run_upper', torch.tensor(0.))
        #self.register_buffer('in_run_lower', torch.tensor(0.))
        self.register_buffer('in_run_scale', torch.tensor(0.))
        self.register_buffer('in_init_state', torch.tensor(0.))
        self.register_parameter('wt_upper', torch.nn.Parameter(torch.tensor(2 ** 2 - 1).float()))
        self.register_parameter('wt_lower', torch.nn.Parameter(torch.tensor((-1) * (2 ** 2)).float()))
        self.register_parameter('wt_alpha', torch.nn.Parameter(torch.tensor(1./4).float()))
        # self.register_buffer('wt_alpha', torch.nn.Parameter(torch.tensor(0).float()))
        self.register_buffer('wt_offset', None)
        self.register_buffer('wt_run_upper', torch.tensor(0.))
        self.register_buffer('wt_run_lower', torch.tensor(0.))
        self.register_buffer('wt_init_state', torch.tensor(0.))
        self.momentum = qconfig['momentum']

    def reset_qparams(self):
        self.in_scale = None
        self.in_offset = None
        self.wt_scale = None
        self.wt_offset = None

    @abstractmethod
    def _forward_func(self, input, weight):
        raise NotImplementedError

    def forward(self, input):
        if self.qconfig['input']['enable']:
            if self.in_init_state == 0:
                in_scale = (torch.max(input) - torch.min(input)) / (self.in_max_val - self.in_min_val)
                #in_lower = torch.min(input)
                # in_scale = 2 * input.detach().abs().mean() / math.sqrt(self.in_max_val)
                #in_lower = torch.min(input)
                self.in_scale.data.copy_(in_scale)
                #self.in_lower.data.copy_(in_lower)
                self.in_run_scale.data.copy_(in_scale)
                #self.in_run_upper = torch.max(input).detach()
                #self.in_run_lower.data.copy_(in_lower)
                #print(self.in_scale, (self.in_run_upper - self.in_run_lower) / (self.in_max_val - self.in_min_val))
                self.in_init_state.fill_(1)

            if self.training:
                g_i = 1 / math.sqrt(input.numel()*self.in_max_val)
                #in_running_lower = self.in_run_lower.mul(1-self.momentum).add(self.momentum * self.in_lower)
                in_running_scale = self.in_run_scale.mul(1-self.momentum).add(self.momentum * self.in_scale)
                #in_running_upper = self.in_scale * (self.in_max_val - self.in_min_val)
                in_running_scale = g_i * in_running_scale + (1 - g_i) * in_running_scale.detach()
                #in_running_scale = self.in_run_scale
                in_running_upper = in_running_scale * (self.in_max_val - self.in_min_val)
                #in_running_lower = g_i * in_running_lower + (1 - g_i) * in_running_lower.detach()
                self.in_run_scale.copy_(in_running_scale.data.detach())
                #self.in_run_lower.copy_(in_running_lower.data.detach())
            else:
                #in_running_lower = self.in_run_lower
                in_running_scale = self.in_run_scale
                in_running_upper = in_running_scale * (self.in_max_val - self.in_min_val)

            input_q = clipping(input, in_running_upper, 0)
            interval = round_pass(input_q/in_running_scale)
            #print("in_scale", self.in_scale)
            input = interval * in_running_scale
        if self.qconfig['weight']['enable']:
            if self.wt_init_state == 0:
                #self.wt_upper.data.copy_(torch.max(self.weight).detach() * 0.9)
                wt_max = 2 * self.weight.detach().abs().mean() * math.sqrt(self.wt_max_val)
                wt_min = -2 * self.weight.detach().abs().mean() * math.sqrt(self.wt_max_val)
                # wt_max = 0.9 * torch.max(self.weight).detach() 
                # wt_min = 0.9 * torch.min(self.weight).detach() 
                print(wt_max, self.weight.max())
                print(wt_min, self.weight.min())
                self.wt_upper.data.copy_(wt_max)
                self.wt_lower.data.copy_(wt_min)
                #self.wt_lower.data.copy_(torch.min(self.weight).detach() * 0.9)
                #print(self.weight.detach().abs().mean() * math.sqrt(self.wt_max_val), self.weight.max())
                #self.wt_upper.data.copy_(w_init * (self.wt_max_val-self.wt_min_val))
                #self.wt_lower.data.copy_(-w_init * (self.wt_max_val-self.wt_min_val) )
                self.wt_run_upper.data.copy_(wt_max)
                self.wt_run_lower.data.copy_(wt_min)
                self.wt_init_state.fill_(1)

            if self.training:
                #print(self.wt_run_upper)
                #print(self.wt_upper)
                #wt_running_upper = self.wt_upper
                #wt_running_lower = self.wt_lower
                g_w = 1 / math.sqrt(self.weight.numel()*self.wt_max_val)
                wt_running_upper = self.wt_run_upper.mul(1-self.momentum).add(self.momentum * self.wt_upper)
                wt_running_lower = self.wt_run_lower.mul(1-self.momentum).add(self.momentum * self.wt_lower)
                wt_running_upper = g_w * wt_running_upper + (1 - g_w) * wt_running_upper.detach()
                wt_running_lower = g_w * wt_running_lower + (1 - g_w) * wt_running_lower.detach()
                self.wt_run_upper.copy_(wt_running_upper.data)
                self.wt_run_lower.copy_(wt_running_lower.data)
            else:
                wt_running_upper = self.wt_run_upper
                wt_running_lower = self.wt_run_lower
            weight_q = clipping(self.weight, wt_running_upper, wt_running_lower)
            delta = (wt_running_upper - wt_running_lower) / (self.wt_max_val - self.wt_min_val)
            interval = floor_pass((weight_q - wt_running_lower)/delta)
            mi = (interval + 0.5) * delta + wt_running_lower
            #print("wt_run_upper", self.wt_run_upper)
            #print("wt_run_lower", self.wt_run_lower)
            weight_q = torch_phi_function(weight_q, mi.detach(), self.wt_alpha, delta)
            #weight_q = torch_phi_function(weight_q, mi.detach(), 0, delta)
            weight_q = sgn(weight_q)
            weight_q = dequantize(weight_q, wt_running_lower, delta, interval)
        return self._forward_func(input, weight_q)

