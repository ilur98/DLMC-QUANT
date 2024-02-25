from typing import Sequence, Dict

import torch
from torch.nn import Module

__all__ = ['DEFAULT_TRACK_LIST', 'get_compute_graph']


def gen_default_track_list():
    from torch.nn import modules
    module_list = modules.__all__.copy()
    module_list.remove("Module")
    module_list.remove("Sequential")
    return tuple(v for k, v in modules.__dict__.items() if k in module_list)
DEFAULT_TRACK_LIST = gen_default_track_list()


def get_compute_graph(
        model:                       Module,
        inputs:      Sequence[torch.Tensor],
        track_types:       Sequence[Module] = DEFAULT_TRACK_LIST,
        style:                          str = "bottom"
) -> Dict[Module, Sequence[Module]]:
    """
    Get compute graph for specific module types.
    :param model: model to get compute graph.
    :param inputs: inputs to feed.
    :param track_types: module types to track.
    :param style: style of graph.
        For "bottom", return bottom graph with a Dict[module -> bottom_modules];
        For "top", return top graph with a Dict[module -> top_modules]
    :return: compute graph.
    """
    # 2nd parameter of `isinstance` function should be a tuple
    track_types = tuple(track_types)

    # Bind grad_fn and tracked module
    fn2module = {}     # grad_fn -> module
    module2fn = {}     # module -> grad_fn
    hook_list = []
    track_modules = []
    def _add_forward_hook(m):
        def _forward_hook(m, x, y):
            fn2module[y.grad_fn] = m
            module2fn[m] = y.grad_fn

        if isinstance(m, track_types):
            hook_list.append(m.register_forward_hook(_forward_hook))
            track_modules.append(m)
    model.apply(_add_forward_hook)
    _ = model(*inputs)
    for h in hook_list:
        h.remove()

    # Do DFS to get bottoms for a module
    def _get_bottoms(m):
        bottoms = []
        visited = []

        def DFS(grad_func):
            # visited or not
            if grad_func is None or grad_func in visited:
                return
            visited.append(grad_func)

            # if bind to a module, record it
            # if not, do recursion to step search
            module = fn2module.get(grad_func)
            if module is None:
                for next_func in getattr(grad_func, "next_functions", []):
                    DFS(next_func[0])
            else:
                bottoms.append(module)

        for next_fn in module2fn[m].next_functions:
            DFS(next_fn[0])

        return bottoms

    # Construct a graph for model and return
    bottom_graph = {m: _get_bottoms(m) for m in track_modules}
    if style == "bottom":    # bottom style, return directly
        return bottom_graph
    elif style == "top":     # top style, swap key and value
        top_graph = {}
        for m, bottoms in bottom_graph.items():
            for b in bottoms:
                top_graph.setdefault(b, []).append(m)
        return top_graph
    else:
        raise ValueError(f"unknown style: {style}")


# ======================== Simple Test ========================
if __name__ == "__main__":
    from torchvision.models import resnet18, mobilenet_v2
    from dlmc.utils.access import mark_modules
    from torch import nn
    model = resnet18()
    print(model)
    print()

    mark_modules(model)
    inputs = [torch.ones(1, 3, 224, 224)]
    # bottoms = get_compute_graph(model, inputs)
    bottoms = get_compute_graph(model, inputs, (nn.Conv2d, ))
    for k, vs in bottoms.items():
        print(f"{k.name} <- {[v.name for v in vs]}")
