from operator import attrgetter
from typing import List, Dict, Callable

import torch

from .count_fn import DEFAULT_COUNT_FN

__all__ = ['count_ops']
__ref__ = 'https://github.com/hey-yahei/OpSummary.MXNet'


def count_ops(
        model:     torch.nn.Module,
        inputs: List[torch.Tensor],
        layers:          List[str],
        count_fn:   List[Callable] = DEFAULT_COUNT_FN
) -> Dict:
    """ Count ops for model
    :param model:
    :param inputs: input tensors to feed model
    :param layers: names of layers to count ops
    :param count_fn: ops count functions
    :return: a dict with (layer_name, ops) as key-value pair
    """
    # hook to record shapes of input and output
    shapes = {}
    def _get_shapes_hook(name):
        def hook(m, x, y):
            shapes[name] = (x[0].shape, y.shape)
        return hook

    # add hook to modules
    hooks = []
    for m_name in layers:
        m = attrgetter(m_name)(model)
        h = m.register_forward_hook(_get_shapes_hook(m_name))
        hooks.append(h)
    # forward to collect shapes
    _ = model(*inputs)
    # remove all hooks
    for h in hooks:
        h.remove()

    # calculate ops for layers
    ops = {}
    for m_name in layers:
        m = attrgetter(m_name)(model)
        for cls, fn in count_fn.items():
            if isinstance(m, cls):
                in_shape, out_shape = shapes[m_name]
                ops[m_name] = fn(m, in_shape, out_shape)
                break

    return ops
