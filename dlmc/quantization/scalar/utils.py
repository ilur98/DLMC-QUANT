def quantize(tensor, scale, offset, min_val, max_val):
    return ((tensor - offset) / (scale + 1e-7)).round().clamp(min_val, max_val)


def dequantize(tensor_q, scale, offset):
    return tensor_q * scale + offset


def emulate_quantize(tensor, scale, offset, min_val, max_val):
    tensor_q = quantize(tensor, scale, offset, min_val, max_val)
    return dequantize(tensor_q, scale, offset)


def get_qrange(signed, n_bits):
    if signed:
        max_val = 2 ** (n_bits - 1) - 1
        min_val = -max_val
    else:
        max_val = 2 ** n_bits - 1
        min_val = 0

    return min_val, max_val

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def cyclic_function(input, k, num_bits):
    g = 2 ** num_bits
    value = input % g
    

    return output