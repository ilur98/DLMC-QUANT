from torch import nn

__all__ = ['DEFAULT_COUNT_FN', 'count_conv2d', 'count_fc']


def count_conv2d(m, x_shape, y_shape):
    cout, cin_g, kw, kh = m.weight.shape
    bs, cin, inw, inh = x_shape
    _, _, outw, outh = y_shape
    g = cin // cin_g

    ops_per_elem = kw * kh * cin_g
    n_elems = bs * outw * outh * cout
    n_ops = ops_per_elem * n_elems
    return n_ops


def count_fc(m, x_shape, y_shape):
    bs = x_shape[0]
    in_fea, out_fea = m.weight.shape

    ops_per_elem = in_fea
    n_elems = bs * out_fea
    n_ops = ops_per_elem * n_elems
    return n_ops


DEFAULT_COUNT_FN = {
    nn.Conv2d: count_conv2d,
    nn.Linear: count_fc
}
