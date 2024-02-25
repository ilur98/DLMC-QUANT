from types import new_class
from matplotlib import scale
from matplotlib.pyplot import sca
import torch
from trainer.loss.loss import l2_loss
from datetime import datetime
import pickle
from .utils import quantize, get_qrange


def get_qparams_output(input, weight, module, qtype, **kwargs):
    quantize_fn = globals()[f"quantize_{qtype}"]
    return quantize_fn(input, weight, module, **kwargs)

def get_qparams_tensor(tensor, qtype, **kwargs):
    # print(f"quantize_{qtype}")
    quantize_fn = globals()[f"quantize_{qtype}"]
    return quantize_fn(tensor, **kwargs)

def quantize_minmax_tensor(tensor, n_bits, signed, allow_offset=True):
    if signed:
        abs_max_val = tensor.abs().max()
        scale = abs_max_val / ((2 ** (n_bits - 1)) - 1)
        offset = torch.tensor(0)
    else:
        min_val = tensor.min()
        # print(min_val)
        if not allow_offset:
            assert (min_val >= 0).all()
            min_val = torch.tensor(0)
        max_val = tensor.max()
        scale = (max_val - min_val) / ((2 ** n_bits) - 1)
        offset = min_val
    return scale, offset

def quantize_l2loss_tensor(tensor, n_bits, signed, allow_offset=True):
    if signed:
        abs_max_val = tensor.abs().max()
        scale = abs_max_val / ((2 ** (n_bits - 1)) - 1)
        offset = torch.tensor(0)
    else:
        min_val = tensor.min()
        # print(min_val)
        if not allow_offset:
            assert (min_val >= 0).all()
            min_val = torch.tensor(0)
        max_val = tensor.max()
        min_loss = 1000
        scale = max_val / ((2 ** (n_bits)) - 1)
        offset = torch.tensor(0)
        t = 0
        for i in range(80):
            new_max_val = (1 - 0.01 * i) * max_val
            new_min_val = (1 - 0.01 * i) * min_val
            new_scale = (new_max_val - new_min_val) / ((2 ** n_bits) - 1)
            # q_tensor = torch.round((tensor - new_min_val) / new_scale)
            # q_tensor = q_tensor.clamp(0, (2 ** n_bits) - 1) * new_scale + new_min_val
            new_offset = torch.round(-new_min_val / new_scale)
            q_tensor = torch.round(tensor / new_scale) + new_offset
            q_tensor = (q_tensor.clamp(0, (2 ** n_bits) - 1) - new_offset) * new_scale
            loss = l2_loss(q_tensor, tensor)
            if loss < min_loss:
                t = i
                min_loss = loss
                scale = new_scale
                offset = new_offset
        print(t, scale, offset)
    return scale, offset
            

def quantize_l2norm_tensor(tensor, n_bits, signed):
    scale, offset = quantize_minmax_tensor(tensor, n_bits, signed, allow_offset=True)
    min_val, max_val = get_qrange(signed, n_bits)

    epsilon = 1e-5
    diff = float('inf')
    while diff > epsilon:
        tensor_q = quantize(tensor, scale, offset, min_val, max_val)
        new_scale = (tensor * tensor_q).sum() / (tensor_q * tensor_q + 1e-7).sum()
        diff = (new_scale - scale).abs() / scale
        scale = new_scale

    return scale, offset

def quantize_l2norm_output(input, weight, module, n_bits, signed, patience=1000):
    output = module._forward_func(input, weight)
    scale, offset = quantize_minmax_tensor(weight, n_bits, signed, allow_offset = True)
    min_val, max_val = get_qrange(signed, n_bits)

    epsilon = 1e-5
    diff = float('inf')
    best_mse = float('inf')
    best_scale = scale
    count = 0
    
    while diff > epsilon:
        if count == patience:
            break
        weight_q = quantize(weight, scale, offset, min_val, max_val)
        output_q = module._forward_func(input, weight_q)
        mse = l2_loss(output, output_q)
        new_scale = (output_q * output).mean(axis=0).sum() / (output_q * output_q + 1e-7).mean(axis=0).sum()
        diff = (new_scale - scale).abs() / scale
        scale = new_scale
        if mse < best_mse:
            best_mse = mse
            best_scale = scale
        count = count + 1
    return best_scale, offset


def _process_channel(tensor, ch_axis):
    new_shape = [1] * len(tensor.shape)
    new_shape[ch_axis] = -1
    n_channels = tensor.shape[ch_axis]
    new_tensor = tensor.transpose(0, ch_axis).reshape(n_channels, -1)

    return new_tensor, new_shape


def quantize_minmax_channel(tensor, n_bits, signed, ch_axis=0, allow_offset=True):
    tensor, new_shape = _process_channel(tensor, ch_axis)

    if signed:
        abs_max_val = tensor.abs().max(dim=1)[0]
        scale = abs_max_val / ((2 ** (n_bits - 1)) - 1)
        offset = torch.zeros_like(scale, device=scale.device)
    else:
        min_val = tensor.min(dim=1)[0]
        # print(min_val)
        if not allow_offset:
            assert (min_val >= 0).all()
            min_val[:] = 0.
        max_val = tensor.max(dim=1)[0]
        scale = (max_val - min_val) / ((2 ** n_bits) - 1)
        offset = min_val

    scale = scale.reshape(new_shape)
    offset = offset.reshape(new_shape)
    return scale, offset

def quantize_minmax_pixel(tensor, n_bits, signed, allow_offset=True):
    if len(tensor.shape) == 4:
        new_shape = [tensor.shape[2], tensor.shape[3]] 
    else:
        new_shape = [tensor.shape[2]]
    oc_num = tensor.shape[0]
    ic_num = tensor.shape[1]
    tensor = tensor.reshape([oc_num, ic_num, -1])
    if signed:
        abs_max_val = tensor.abs().max(dim=0)[0]
        abs_max_val = abs_max_val.max(dim=0)[0]
        scale = abs_max_val / ((2 ** (n_bits - 1)) - 1)
        offset = torch.zeros_like(scale, device=scale.device)
    else:
        min_val = tensor.abs().min(dim=0)[0]
        min_val = min_val.min(dim = 0)[0]
        max_val = tensor.abs().max(dim=0)[0]
        max_val = max_val.max(dim = 0)[0]
        if not allow_offset:
            assert(min_val >= 0).all()
            min_val[:] = 0.
        scale = (max_val - min_val) / ((2 ** n_bits) - 1)
        offset = min_val
    scale = scale.reshape(new_shape)
    offset = offset.reshape(new_shape)
    return scale, offset

def quantize_l2loss_channel(tensor, n_bits, signed, ch_axis=0):
    tensor, new_shape = _process_channel(tensor, ch_axis)
    scale, offset = quantize_minmax_channel(tensor, n_bits, signed, ch_axis=0, allow_offset=True)
    min_val = offset
    max_val = offset + scale * ((2 ** n_bits) - 1)
    g = tensor.shape
    for c in range(g[0]):
        min_loss = 1000
        for i in range(80):
            # print(i)
            new_min_val = (1 - 0.01 * i) * min_val[c]
            new_max_val = (1 - 0.01 * i) * max_val[c]
            new_scale = (new_max_val - new_min_val) / ((2 ** n_bits) - 1)
            # new_offset = new_min_val
            # tensor_q = (tensor[c] - new_offset) / new_scale
            # tensor_q = tensor_q.round().clamp(0, (2 ** n_bits) - 1) * new_scale + new_offset
            new_offset = torch.round(-new_min_val / new_scale)
            tensor_q = torch.round(tensor[c] / new_scale)
            tensor_q = (tensor_q + new_offset).clamp(0, (2 ** n_bits) - 1) 
            tensor_q = (tensor_q - new_offset) * new_scale
            # print(tensor[c].shape, tensor_q.shape)
            loss = l2_loss(tensor[c].view(1, -1), tensor_q.view(1, -1))
            if min_loss > loss:
                scale[c] = new_scale
                offset[c] = new_offset
                min_loss = loss

    return scale.reshape(new_shape), offset.reshape(new_shape)

def quantize_l2norm_channel(tensor, n_bits, signed, ch_axis=0):
    tensor, new_shape = _process_channel(tensor, ch_axis)
    scale, offset = quantize_minmax_channel(tensor, n_bits, signed, ch_axis=0, allow_offset=True)
    min_val, max_val = get_qrange(signed, n_bits)

    epsilon = 1e-5
    diff = float('inf')
    while diff > epsilon:
        tensor_q = quantize(tensor, scale, offset, min_val, max_val)
        new_scale = (tensor * tensor_q).sum(axis=1) / (tensor_q * tensor_q + 1e-7).sum(axis=1)
        new_scale = new_scale.reshape(scale.shape)
        diff = ((new_scale - scale) ** 2).sum().sqrt() / (scale ** 2).sum().sqrt()
        scale = new_scale
        # print(diff)

    scale = scale.reshape(new_shape)
    offset = offset.reshape(new_shape)
    return scale, offset

def quantize_l2norm_pixel(tensor, n_bits, signed, patience=1000):
    if len(tensor.shape) == 4:
        new_shape = [tensor.shape[2], tensor.shape[3]] 
    elif len(tensor.shape) == 3:
        new_shape = [tensor.shape[2]]
    else:
        new_shape = [1]
    oc_num = tensor.shape[0]
    ic_num = tensor.shape[1]
    tensor = tensor.reshape((oc_num, ic_num, -1))
    scale, offset = quantize_minmax_pixel(tensor, n_bits, signed)
    min_val, max_val = get_qrange(signed, n_bits)
    
    epsilon = 1e-5
    diff = float('inf')
    best_mse = float('inf')
    count = 0
    while diff > epsilon:
        if count == patience:
            break
        tensor_q = emulate_quantize(tensor, scale, offset, min_val, max_val)
        new_scale = (tensor * tensor_q).sum(axis=(0,1)) / (tensor_q * tensor_q + 1e-7).sum(axis=(0,1))
        mse = l2_loss(tensor, tensor_q)
        new_scale = new_scale.reshape(scale.shape)
        diff = ((new_scale - scale) ** 2).sum().sqrt() / (scale ** 2).sum().sqrt()
        if best_mse > mse:
            best_mse = best_mse
            best_scale = scale
        scale = new_scale
        count = count+1

    scale = best_scale.reshape(new_shape)
    offset = offset.reshape(new_shape)
    return scale, offset

def quantize_l2norm_output_channel(input, weight, module, n_bits, signed, ch_axis=0, patience=1000):
    tensor, new_shape = _process_channel(weight, ch_axis)
    output = module._forward_func(input, weight)
    batch   = output.shape[0]
    channel = output.shape[1]
    output = output.reshape(batch, channel, -1)
    scale, offset = quantize_minmax_channel(tensor, n_bits, signed, ch_axis=0, allow_offset=True)
    scale = scale.reshape(new_shape)
    offset = offset.reshape(new_shape)
    min_val, max_val = get_qrange(signed, n_bits)
    count = 0
    epsilon = 1e-5
    diff = float('inf')
    best_mse = float('inf')
    best_scale = scale
    mse_list = []
    while diff > epsilon:
        if count == patience:
            break
        weight_q = quantize(weight, scale, offset, min_val, max_val)
        output_q = module._forward_func(input, weight_q)
        output_q = output_q.reshape(batch, channel, -1)
        new_scale = (output * output_q).sum(axis=(0,2)) / (output_q * output_q + 1e-7).sum(axis=(0,2))
        #print(new_scale.size())
        new_scale = new_scale.reshape(scale.shape)
        mse = l2_loss(output, output_q)
        diff = ((new_scale - scale) ** 2).sum().sqrt() / (scale ** 2).sum().sqrt()
        if mse < best_mse:
            best_mse = mse
            best_scale = scale
        mse_list.append(mse)
        scale = new_scale
        count = count + 1

    # run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    # filename = "E:/data/dlmc/saved/diff/" + run_id + ".txt"
    # with open(filename, "ab+") as f:
    #     pickle.dump(mse, f)

    best_scale = best_scale.reshape(new_shape)
    return best_scale, offset
