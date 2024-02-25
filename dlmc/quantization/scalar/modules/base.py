from abc import ABCMeta, abstractmethod
from fnmatch import fnmatch
import torch
import math
from torch.nn import Module
from pathlib import PosixPath
from ..ops import get_qparams_tensor, get_qparams_output
from ..utils import emulate_quantize, get_qrange, grad_scale, round_pass
from ..modules.function import *

class QBase(Module):
    __metaclass__ = ABCMeta

    qconfig: dict
    wt_min_val: int
    wt_max_val: int
    wt_scale: torch.nn.Parameter
    wt_offset: torch.TensorType
    in_min_val: int
    in_max_val: int
    in_scale: torch.nn.Parameter
    in_offset: torch.TensorType

    def __init__(self, qconfig: dict = None):
        super(QBase, self).__init__()
        self.initialize(qconfig)

    def initialize(self, qconfig):
        if 'channel' in qconfig['input']['type']:
            qconfig['input']['args']['ch_axis'] = 1
        self.qconfig = qconfig

        self.wt_min_val, self.wt_max_val = get_qrange(
            qconfig['weight']['args']['signed'],
            qconfig['weight']['args']['n_bits']
        )
        self.in_min_val, self.in_max_val = get_qrange(
            qconfig['input']['args']['signed'],
            qconfig['input']['args']['n_bits']
        )
        #in_scale = None
        #wt_scale = None 
        #if fnmatch(self.qconfig['input']['type'], 'LSQ'):
        #    #in_scale = torch.nn.Parameter(2 * input.detach().abs().mean() / math.sqrt(self.in_max_val))
        #    in_scale = torch.nn.Parameter(torch.ones(1))
        #    wt_scale = torch.nn.Parameter(torch.ones(1))
            #wt_scale = torch.nn.Parameter(2 * self.weight.detach().abs().mean() / math.sqrt(self.wt_max_val))
        self.register_parameter('in_scale', torch.nn.Parameter(torch.ones(1)))
        #self.in_scale = None
        self.register_buffer('in_offset', None)
        self.register_buffer('in_init_state', torch.zeros(1))
        self.register_parameter('wt_scale', torch.nn.Parameter(torch.ones(1)))
        #self.wt_scale = None
        self.register_buffer('wt_offset', None)
        self.register_buffer('wt_init_state', torch.zeros(1))

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
            #print(self.in_scale)
            #if self.in_init_state == 0 and fnmatch(self.qconfig['input']['type'], 'LSQ'):
            #    self.in_scale.data.copy_(2 * input.detach().abs().mean() / math.sqrt(self.in_max_val))
            #    self.in_init_state.fill_(1)
            
            #if self.in_scale is None:
            #    scale, self.in_offset = get_qparams_tensor(
            #        input.detach(),
            #        qtype=self.qconfig['input']['type'],
            #        **self.qconfig['input']['args']
            #    )
            #    self.in_scale.data.copy_(scale)
            #    self.in_init_state.fill_(1)
            if self.in_init_state == 0:
                if fnmatch(self.qconfig['input']['type'], 'LSQ'):
                    self.in_scale.data.copy_(2 * input.detach().abs().mean() / math.sqrt(self.in_max_val))
                    self.in_offset = torch.zeros(1, device=torch.device('cuda'))
                    self.in_init_state.fill_(1)
                else:
                    scale, self.in_offset = get_qparams_tensor(
                     input.detach(),
                     qtype=self.qconfig['input']['type'],
                     **self.qconfig['input']['args']
                    )
                    self.in_scale.data.copy_(scale)
                    self.in_init_state.fill_(1)
            
            g_i = 1 / math.sqrt(input.numel()*self.in_max_val)
            i_scale = grad_scale(self.in_scale, g_i)
            #input_q = emulate_quantize(input.detach(), self.in_scale, self.in_offset,
            #                           self.in_min_val, self.in_max_val)
            #noise = input_q - input
            #input_q = input + noise.detach()
            input = round_pass(((input - self.in_offset) / i_scale).clamp(self.in_min_val, self.in_max_val)) * i_scale + self.in_offset
            
            #input = FunLSQ.apply(input, self.in_scale, self.in_offset, self.in_min_val, self.in_max_val, g_i)

        if self.qconfig['weight']['enable']:
            if self.wt_init_state == 0:
                if fnmatch(self.qconfig['weight']['type'], '*output*'):
                    scale, self.wt_offset = get_qparams_output(
                        input.detach(),
                        self.weight.detach(),
                        self,
                        qtype=self.qconfig['weight']['type'],
                        **self.qconfig['weight']['args']
                    )
                    self.wt_scale.data.copy_(scale)
                    self.wt_init_state.fill_(1)
                elif fnmatch(self.qconfig['weight']['type'], 'LSQ'):
                    self.wt_scale.data.copy_(2 * self.weight.detach().abs().mean() / math.sqrt(self.wt_max_val))
                    self.wt_offset = torch.zeros(1, device=torch.device('cuda'))
                    self.wt_init_state.fill_(1)
                else:
                    scale, self.wt_offset = get_qparams_tensor(
                        self.weight.detach(),
                        qtype=self.qconfig['weight']['type'],
                        **self.qconfig['weight']['args']
                    )
                    self.wt_scale.data.copy_(scale)
                    self.wt_init_state.fill_(1)
                    
            g_w = 1 / math.sqrt(self.weight.numel()*self.wt_max_val)
            w_scale = grad_scale(self.wt_scale, g_w)
            weight = round_pass(((self.weight -self.wt_offset) / w_scale).clamp(self.wt_min_val, self.wt_max_val)) * w_scale + self.wt_offset
            #weight_q = emulate_quantize(self.weight.detach(), w_scale, self.wt_offset,
            #                            self.wt_min_val, self.wt_max_val)
            #noise = weight_q - self.weight
            #weight = self.weight + noise.detach()
            #weight = FunLSQ.apply(self.weight, self.wt_scale, self.wt_offset, self.wt_min_val, self.wt_max_val, g_w)
        #print(self.wt_scale, self.in_scale)
        return self._forward_func(input, weight)