from operator import attrgetter
import copy
from typing import Dict
import torch
from torch import nn
from torch.nn.modules.linear import Linear

from .merge_bn import merge_bn
from ..quantization.scalar import modules as qnn
from ..quantization.scalar import BitMixer as BM
from ..quantization.scalar import RootQ as RQ
from ..quantization.scalar import MetaQuant as MQ
from ..quantization.scalar import FSPTQuant as FSPQ
from timm.models.resnet import BasicBlock
from .access import get_layers, attrsetter

__all__ = ['quantize_model']

MODULE_MAPPING = {
    nn.Conv2d: qnn.QConv2d,
    nn.Linear: qnn.QLinear
}

ROOTQ_MAPPING = {
    nn.Conv2d: RQ.RootQConv2d,
    nn.Linear: RQ.RootQLinear
}

BITMIXER_MAPPING = {
    nn.Conv2d: BM.BitMixerConv2d,
    nn.Linear: BM.BitMixerLinear
}

METAQUATN_MAPPING = {
    nn.Conv2d: MQ.MetaQConv2d,
    nn.Linear: MQ.MetaQLinear
}

FSPTQUANT_MAPPING = {
    nn.Conv2d: FSPQ.FSPTQConv2d,
    nn.Linear: FSPQ.FSPTQLinear
}

def _override_options(
        dst_config: Dict,
        src_config: Dict = None
) -> Dict:
    if src_config is None:
        return dst_config

    dst_config = copy.deepcopy(dst_config)
    if 'type' in src_config:
        dst_config['type'] = src_config['type']
    if 'enable' in src_config:
        dst_config['enable'] = src_config['enable']
    if 'args' in src_config:
        dst_config['args'].update(src_config['args'])
    return dst_config


def quantize_model(
        model: nn.Module,
        config:     Dict,
        logger,
        quantization_type: str = None,
        **kwargs
) -> None:
    """
    Easy API to quantize a model
    :param model: model to quantize
    :param config: quantization configuration
    :param logger: logger to write
        TODO: maybe logger should be taken away from `quantize_model` API
    """
    default_weight_config = config['weight']
    default_input_config = config['input']
    default_momentum_config = 0.1
    if quantization_type == "BitMixer":
        print("doing bitmixer")
        MAPPING_DICT = BITMIXER_MAPPING
        #merge_bn(model, bitmixer_func=True)
        #print(model)
        bn_layers = get_layers(model, filter_types = tuple([nn.BatchNorm2d]))
        #print(bn_layers)
        for layer in bn_layers:
            module = attrgetter(layer)(model)
            #dict1 = module.__dict__
            #BatchNorms = BM.BitMixerBatchNorm(dict1['num_features'], dict1['eps'], dict1['momentum'], dict1['affine'])
            BatchNorms = BM.BitMixerBatchNorm(module.num_features, module.eps, module.momentum, module.affine)
            attrsetter(layer)(model, BatchNorms)
            logger.info("change the batchnorm{} to batchnorms".format(layer))

    elif quantization_type == "RootQ":
        MAPPING_DICT = ROOTQ_MAPPING
        default_momentum_config = config['momentum']
    elif quantization_type == "MetaQ":
        MAPPING_DICT = METAQUATN_MAPPING
    elif quantization_type == "FSPTQ":
        MAPPING_DICT = FSPTQUANT_MAPPING
    else:
        MAPPING_DICT = MODULE_MAPPING

    all_layers = get_layers(model, filter_types=tuple(MAPPING_DICT.keys()))
    all_blocks = get_layers(model, filter_types=BasicBlock)
    print(all_blocks)
    exclude_layers = []
    for regexp in config['exclude_layers']:
        excludes = get_layers(model, filter_regexp=regexp)
        exclude_layers.extend(excludes)
    quantized_layers = [l for l in all_layers if l not in exclude_layers]

    override_options = {}
    for opt in config['override_options']:
        for regexp in opt['layers']:
            layers = get_layers(model, filter_regexp=regexp)
            for l in layers:
                assert l not in override_options
                override_options[l] = opt['options']
    for layer in quantized_layers:
        module = attrgetter(layer)(model)

        weight_config = default_weight_config
        input_config = default_input_config
        momentum_config = default_momentum_config
        if layer in override_options:
            weight_config = _override_options(weight_config, override_options[layer].get("weight", None))
            input_config = _override_options(input_config, override_options[layer].get("input", None))
        config = {"input": input_config, "weight": weight_config, "momentum":momentum_config}
        print(module)
        new_type = MAPPING_DICT[type(module)]
        module_q = new_type.__new__(new_type)
        module_q.__dict__.update(module.__dict__)
        module_q.initialize(config)
        if quantization_type == "MetaQ":
            module_q.init_meta(**kwargs)
        attrsetter(layer)(model, module_q)

        #logger.info("Quantize module {} with method <input: {}> <weight: {}>".format(
        #    layer, config['input'], config['weight']
        #))
        print("Quantize module {} with method <input: {}> <weight: {}>".format(
            layer, config['input'], config['weight']
        ))