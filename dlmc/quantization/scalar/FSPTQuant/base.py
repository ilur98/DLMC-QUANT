from abc import ABCMeta, abstractmethod
from copy import deepcopy
from fnmatch import fnmatch
from einops import rearrange
from matplotlib.pyplot import sca
import torch
import math
from torch.nn import Module
from pathlib import PosixPath
from ..ops import get_qparams_tensor, get_qparams_output
from ..utils import emulate_quantize, get_qrange, grad_scale, round_pass, floor_pass
from ..modules.function import *

class FSPTQBase(Module):
    __metaclass__ = ABCMeta

    qconfig: dict
    wt_min_val: int
    wt_max_val: int
    # wt_scale: torch.nn.Parameter
    # wt_offset: torch.TensorType
    in_min_val: int
    in_max_val: int
    in_scale: torch.nn.Parameter
    in_offset: torch.TensorType

    def __init__(self, qconfig: dict = None):
        super(FSPTQBase, self).__init__()
        self.initialize(qconfig)
        shape = self.weight.shape()
        

    def initialize(self, qconfig):
        self.qconfig = qconfig
        self.train_module = 0
        self.wt_min_val, self.wt_max_val = get_qrange(
            qconfig['weight']['args']['signed'],
            qconfig['weight']['args']['n_bits']
        )
        self.in_min_val, self.in_max_val = get_qrange(
            qconfig['input']['args']['signed'],
            qconfig['input']['args']['n_bits']
        )

        channel = self.weight.shape[0]
        self.register_parameter('in_scale', torch.nn.Parameter(torch.ones(1)))
        self.register_buffer('in_offset',  torch.zeros(1, device=torch.device('cuda')))
        self.register_buffer('in_init_state', torch.zeros(1))
        if len(self.weight.shape) == 4:
            self.register_parameter('wt_scale', torch.nn.Parameter(torch.ones(channel, 1, 1, 1)))
            self.register_buffer('wt_offset', torch.nn.Parameter(torch.ones(channel, 1, 1, 1)))
        else:
            self.register_parameter('wt_scale', torch.nn.Parameter(torch.ones(channel, 1)))
            self.register_buffer('wt_offset', torch.nn.Parameter(torch.ones(channel, 1)))
        self.register_buffer('wt_init_state', torch.zeros(1))
        self.register_buffer('org_weight', self.weight.clone().detach())
        self.act_quant = self.qconfig['input']['enable']
        self.wt_quant = self.qconfig['weight']['enable']
        self.soft_target = True
        if self.qconfig['weight']['recon_type'] in ["adaround", "dist_recon"]:
            self.register_parameter('alpha', torch.nn.Parameter(torch.ones_like(self.weight)))
            self.gamma, self.zeta = -0.1, 1.1
            self.beta = 2/3

    @abstractmethod
    def _forward_func(self, input, weight):
        raise NotImplementedError

    def init_alpha(self):
        # print(self.weight.shape, self.wt_scale.shape)
        # w_floor = torch.floor((self.weight.detach() - self.wt_offset) / self.wt_scale) 
        # rest = (self.weight.detach() - self.wt_offset)/ self.wt_scale - w_floor
        w_floor = torch.floor(self.weight.detach() / self.wt_scale)
        rest = self.weight.detach() / self.wt_scale - w_floor
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
        self.alpha.data.copy_(alpha)

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def change_quant_state(self, wt_state, act_state):
        self.wt_quant = wt_state
        self.act_quant = act_state

    def reinit_parameters(self):
        self.in_init_state.fill_(0)
        self.wt_init_state.fill_(0)

    def debug(self):
        q_weight = floor_pass(((self.weight - self.wt_offset) / self.wt_scale))
        q_weight = q_weight + (self.alpha >= 0).float()
        q_weight2 = round_pass(((self.weight -self.wt_offset) / self.wt_scale))
        # print((q_weight2 == q_weight.round()).sum() / q_weight.numel())

    def forward(self, input):
        # print(self.act_quant, self.wt_quant)
        if self.act_quant:
            if self.in_init_state == 0:
                scale, self.in_offset = get_qparams_tensor(
                    input.detach(),
                    qtype=self.qconfig['input']['type'],
                    **self.qconfig['input']['args']
                )
                self.in_scale.data.copy_(scale)
                self.in_init_state.fill_(1)

            # q_input = round_pass(((input - self.in_offset) /self.in_scale).clamp(self.in_min_val, self.in_max_val)) * self.in_scale + self.in_offset
            q_input = (round_pass(input / self.in_scale) + self.in_offset).clamp(self.in_min_val, self.in_max_val)
            q_input = (q_input - self.in_offset) * self.in_scale

        if self.wt_quant:
            if self.wt_init_state == 0:
                scale, self.wt_offset = get_qparams_tensor(
                        self.weight.detach(),
                        qtype=self.qconfig['weight']['type'],
                        **self.qconfig['weight']['args']
                    )
                # scale, self.wt_offset = get_qparams_output(
                #             input,
                #             self.weight.detach(),
                #             module=self,
                #             qtype=self.qconfig['weight']['type'],
                #             **self.qconfig['weight']['args']
                # )
                # print(scale.min())
                # print(scale.shape, self.wt_scale.shape)
                # print(scale, (self.weight.max(dim=1)[0] - self.weight.min(dim=1)[0]) / (self.wt_max_val - self.wt_min_val))
                # print(self.weight.max(dim=(0, 2, 3)) , self.weight.min(dim=(0, 2, 3)))
                self.wt_scale.data.copy_(scale+1e-6)
                if self.qconfig['weight']['recon_type'] == 'adaround':
                    self.init_alpha()
                elif self.qconfig['weight']['recon_type'] == 'dist_recon':
                    self.init_wt_alpha()
                self.wt_init_state.fill_(1)
                
            if self.qconfig['weight']['recon_type'] == 'adaround':
                q_weight = torch.floor(self.weight / self.wt_scale)
                if self.training:
                    q_weight = q_weight + self.get_soft_targets()
                else:
                    q_weight = q_weight + (self.alpha >= 0).float()
            elif self.qconfig['weight']['recon_type'] == 'dist_recon':
                q_weight = torch.floor()
                if self.soft_target:
                    q_weight = q_weight + self.get_soft_targets()
                else:
                    q_weight = q_weight + (self.alpha >= 0).float()
            else:
                q_weight = round_pass(self.weight / self.wt_scale)
            
            weight = q_weight.clamp(self.wt_min_val, self.wt_max_val)
            weight = weight * self.wt_scale
            # print((self.weight - q_weight).abs().pow(2).mean())
            out = self._forward_func(q_input, weight)
        else:
            out = self._forward_func(q_input, self.weight)

        
        return out