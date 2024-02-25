import torch
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)


def native_cross_entropy_loss(t1, t2):
    cross_entropy = -(t2 * (t1 + 1e-7).log()).sum(axis=1)
    return cross_entropy.mean()


def kl_loss(t1, t2):
    kl = (t2 * ((t2 + 1e-7) / (t1 + 1e-7)).log()).sum(axis=1)
    return kl.mean()


def l2_loss(t1, t2):
    l2 = ((t1 - t2) ** 2).sum(axis=1)
    return l2.mean()

def Kurt(x):
    tmp = ((x - x.mean()) / x.std()) ** 4

    return tmp.mean()

def kutosis_loss(weights, target):
    total = 0
    for weight in weights:
        total += (Kurt(weight) - target) ** 2
    
    return total / len(weights)

def smoothlabel_ce_loss(output, target, eps=0.3, reduction='mean'):
    num_classes = output.shape[1]
    one_hot_label = output.data.clone().zero_().scatter_(1, target.unsqueeze(1), 1)
    target = (1 - eps) * one_hot_label + (eps / (num_classes - 1)) * (1 - one_hot_label)

    x = F.log_softmax(output, dim=1)
    if reduction == 'mean':
        loss = -torch.sum(torch.sum(x * target, dim=1)) / x.size(0)
    elif reduction == 'sum':
        loss = -torch.sum(torch.sum(x * target, dim=1))
    elif reduction == 'none':
        loss = -torch.sum(x * target, dim=1)
    else:
        raise ValueError('Unknown reduction type {}.'.format(reduction))
    return loss