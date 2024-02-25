# -*- coding: utf-8 -*-

import re
from operator import itemgetter, attrgetter
from typing import List, Iterable

from torch import nn

__all__ = ['attrsetter', 'get_layers', 'mark_modules']


def attrsetter(*items):
    def resolve_attr(obj, attr):
        attrs = attr.split(".")
        head = attrs[:-1]
        tail = attrs[-1]

        for name in head:
            obj = getattr(obj, name)
        return obj, tail

    def g(obj, val):
        for attr in items:
            resolved_obj, resolved_attr = resolve_attr(obj, attr)
            setattr(resolved_obj, resolved_attr, val)

    return g


def get_layers(
        model:                  nn.Module,
        filter_regexp:                str = "(.*?)",
        filter_types: Iterable[nn.Module] = None
) -> List[str]:
    """
    Get all layer names according to filter
    :param model: model
    :param filter_regexp: regular expression to filter names
    :param filter_types: module types to filter
    :return: layer names in model according to filter
    """
    # get all parameter names
    all_layers = map(itemgetter(0), model.named_parameters())

    # remove biases
    all_layers = filter(lambda x: "bias" not in x, all_layers)

    # remove .weight in all other names (or .weight_orig is spectral norm)
    all_layers = map(lambda x: x.replace(".weight_orig", ""), all_layers)
    all_layers = map(lambda x: x.replace(".weight", ""), all_layers)

    # filter layers by name
    filter_regexp = "(module\\.)?" + "(" + filter_regexp + ")"
    r = re.compile(filter_regexp)
    all_layers = list(filter(r.match, all_layers))

    # filter layers by type
    if filter_types is not None:
        all_layers = [l for l in all_layers if isinstance(attrgetter(l)(model), filter_types)]

    return all_layers


def mark_modules(
        module:    nn.Module,
        recursion:      bool = True,
        ancestors: List[str] = []
) -> None:
    """
    Set attribute `name` for every children modules
    :param module: module to mark
    :param recursion: if True, mark children modules recursively
    :param ancestors: split names of ancestors
    """
    for name, m in module._modules.items():
        full_name = ancestors + [name]
        m.name = ".".join(full_name)
        if recursion:
            mark_modules(m, recursion, full_name)
