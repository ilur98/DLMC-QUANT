from copy import deepcopy
from operator import attrgetter
from types import FunctionType

import torch
from torch.nn import Conv2d, BatchNorm2d, Identity, Parameter, Module
from dlmc.quantization.scalar.BitMixer import BitMixerBatchNorm, BitMixerSwitchableBatchNorm
from .access import get_layers, attrsetter

__all__ = ['DEFAULT_BN_MAPPING_FN', 'DEFAULT_CONV_MAPPING_FN', 'merge_bn']


def DEFAULT_CONV_MAPPING_FN(bn_name: str) -> str or None:
    # case1: layer1.conv1.1 -> layer1.conv1.0
    # case2: layer1.bn1 -> layer1.conv1
    name_split = bn_name.split(".")
    parent_name = name_split[:-1]
    base_name = name_split[-1]
    if base_name.isdecimal():
        ret_split = parent_name + [str(int(base_name)-1)]
    elif "bn" in base_name:
        ret_split = parent_name + [base_name.replace('bn', 'conv')]
    else:
        return None
    ret = ".".join(ret_split)
    return ret


def DEFAULT_BN_MAPPING_FN(conv_name: str) -> str or None:
    # case1: layer1.conv1.0 -> layer1.conv1.1
    # case2: layer1.conv1 -> layer1.bn1
    name_split = conv_name.split(".")
    parent_name = name_split[:-1]
    base_name = name_split[-1]
    if base_name.isdecimal():
        ret_split = parent_name + [str(int(base_name)+1)]
    elif "conv" in base_name:
        ret_split = parent_name + [base_name.replace('conv', 'bn')]
    else:
        return None
    ret = ".".join(ret_split)
    return ret


def merge_bn(
        model:            Module,
        mapping_fn: FunctionType = DEFAULT_CONV_MAPPING_FN,
        inplace:            bool = False,
        allow_missing:      bool = False,
        bitmixer_func:      bool = False,
) -> Module:
    """
    Try to merge `BatchNorm2d` layer into `Conv2d` layer
    :param model: model to merge bn
    :param mapping_fn: mapping function who input bn name and output conv name
    :param inplace: merge bn inplace or not
    :param allow_missing: if False, all `BatchNorm2d` should be matched by some `Conv2d`
    :return: bn-merged model
    """
    if inplace:
        model = deepcopy(model)

    # Get all `Conv2d` and `BatchNorm2d`
    all_layers = get_layers(model, filter_types=(Conv2d, BatchNorm2d))
    for layer in all_layers:
        module = attrgetter(layer)(model)
        if isinstance(module, BatchNorm2d):
            # Get matched `Conv2d` layer
            map_name = mapping_fn(layer)
            if map_name is None or map_name not in all_layers:
                info_str = f"[MergeBN] Could not find Conv2d that match {layer}"
                if allow_missing:
                    print(info_str)
                else:
                    raise ValueError(info_str)
            dst_module = attrgetter(map_name)(model)

            # Get initial parameters from `BatchNorm2d`
            num_features = module.num_features
            eps = module.eps
            momentum = module.momentum
            affine = module.affine

            # Get related parameters from `BatchNorm2d`
            gamma = module.weight.data
            beta = module.bias.data
            mean = module.running_mean.data
            var = module.running_var.data + 1e-7

            # Get related parameters from `Conv2d`
            weight = dst_module.weight.data
            w_shape = weight.shape
            cout = w_shape[0]
            # Ensure that `bias` exists
            if dst_module.bias is None:
                dst_module.bias = Parameter(torch.zeros(cout, device=weight.device))
            bias = dst_module.bias.data

            # Update parameters for `Conv2d`
            bias[:] = gamma * (bias - mean) / var.sqrt() + beta
            weight[:] = (weight.reshape(cout, -1) * gamma.reshape(-1, 1) / var.sqrt().reshape(-1, 1)).reshape(w_shape)

            # Replace `BatchNorm2d` with `Identity`
            name_split = layer.split(".")
            parent_name = ".".join(name_split[:-1])
            base_name = name_split[-1]
            parent = attrgetter(parent_name)(model) if parent_name != "" else model
            if bitmixer_func is True:
                attrsetter(base_name)(parent, BitMixerBatchNorm(num_features, eps, momentum, affine))
            else:
                attrsetter(base_name)(parent, Identity())

    return model
