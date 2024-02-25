import argparse
import torch
import time
import numpy as np

import sys
sys.path.append(".")
import data_loader.data_loaders as module_data
import trainer.loss as module_loss
import model as module_arch
from parse_config import ConfigParser
from utils import read_yaml
from torch.utils.data.distributed import DistributedSampler
from base import BaseDataLoader


class ProgressBar(object):
    def __init__(self, total_cnt, width=50):
        self.pointer = 0
        self.total_cnt = total_cnt
        self.width = width

    def __call__(self, x, **kwargs):
        # x in percent
        self.pointer = int(self.width * (x / self.total_cnt))
        len_sum = len(str(self.total_cnt))
        len_cur = len(str(x))
        ret = "\r|" + "#" * self.pointer + "-" * (self.width-self.pointer) + "|" + \
            " " * (len_sum - len_cur + 2) + "{}/{}".format(x, self.total_cnt)
        for (k, v) in kwargs.items():
            ret += "  " + k + ": " + str(v)
        return ret


class _MyDataset(torch.utils.data.DataLoader):
    def __init__(self):
        pass

    def __len__(self):
        return 1281167

    def __getitem__(self, index):
        img = torch.randn(3, 224, 224)
        time.sleep(0.001)
        img = (img - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return img, int(1)


class EmptyLoader(BaseDataLoader):
    def __init__(self, dataset, data_dir, batch_size, shuffle=True, training=True, drop_last=False,
                 validation_split=0.0, num_workers=1, image_size=224):
        self.image_size = image_size

        super().__init__(dataset, batch_size, shuffle, drop_last, validation_split, num_workers)


def _prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
              "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def main(config):
    device, device_ids = _prepare_device(config['n_gpu'])
    if config['dataloader']['type'] == 'Empty':
        dataloader = EmptyLoader(_MyDataset(), **dict(config['dataloader']['args']))
    else:
        dataloader = getattr(module_data, config['dataloader']['type'])(**dict(config['dataloader']['args']))

    if dataloader.drop_last:
        total_cnt = dataloader.batch_size * (len(dataloader) if steps is None else steps)
    else:
        total_cnt = len(dataloader.dataset) if steps is None else dataloader.batch_size * steps

    test_models = config['arch']
    res = dict()
    for m in test_models:
        model_name = m.get('name', m['type'])
        model = getattr(module_arch, m['type'])(**dict(m['args'])).to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = getattr(torch.optim, config['optimizer']['type'])(trainable_params, **dict(config['optimizer']['args']))

        total_time = 0.0
        total_ips = 0.0
        for ri in range(repeat):
            print("\nModel: %s,    Round %d" % (model_name, ri + 1))
            progress_bar = ProgressBar(total_cnt=total_cnt)
            ips = benchmark(model, dataloader, optimizer, criterion, progress_bar, train, steps, device)
            epc_time = len(dataloader.dataset) / ips
            total_time, total_ips = total_time + epc_time, total_ips + ips

        res[model_name] = {'epc_time': total_time / repeat, 'ips': total_ips / repeat}
        print('\n', model_name, res[model_name])

    print()
    for k, v in res.items():
        print("%s: epc_time = %.2fs, ips = %.1f" % (k, v['epc_time'], v['ips']))


def main_ddp(config):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if config['dataloader']['type'] == 'Empty':
        dataloader = EmptyLoader(_MyDataset(), **dict(config['dataloader']['args']))
    else:
        dataloader = getattr(module_data, config['dataloader']['type'])(**dict(config['dataloader']['args']))
    ddp_sampler = torch.utils.data.distributed.DistributedSampler(dataloader.dataset)
    args = {
        'dataset': dataloader.dataset,
        'batch_size': dataloader.batch_size,
        'sampler': ddp_sampler,
        'num_workers': dataloader.num_workers,
        'drop_last': dataloader.drop_last
    }
    dataloader = torch.utils.data.DataLoader(**args)

    if dataloader.drop_last:
        total_cnt = dataloader.batch_size * (len(dataloader) if steps is None else steps) * config['n_gpu']
    else:
        total_cnt = len(dataloader.dataset) if steps is None else dataloader.batch_size * steps * config['n_gpu']

    n_gpu = config['n_gpu']
    test_models = config['arch']
    res = dict()
    for m in test_models:
        model_name = m.get('name', m['type'])
        model = getattr(module_arch, m['type'])(**dict(m['args'])).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = getattr(torch.optim, config['optimizer']['type'])(trainable_params, **dict(config['optimizer']['args']))

        total_time = 0.0
        total_ips = 0.0
        for ri in range(repeat):
            if local_rank == 0:
                print("\nModel: %s,    Round %d" % (model_name, ri + 1))
            progress_bar = ProgressBar(total_cnt=total_cnt)
            ips = benchmark(model, dataloader, optimizer, criterion, progress_bar, train, steps, device, ddp_gpu=n_gpu)
            epc_time = len(dataloader.dataset) / ips
            total_time, total_ips = total_time + epc_time, total_ips + ips

        res[model_name] = {'epc_time': total_time / repeat, 'ips': total_ips / repeat}
        if local_rank == 0:
            print('\n', model_name, res[model_name])

    if local_rank == 0:
        print()
        for k, v in res.items():
            print("%s: epc_time = %.2fs, ips = %.1f" % (k, v['epc_time'], v['ips']))


def benchmark(model, dataloader, optimizer, criterion, progress_bar, train, steps, device, ddp_gpu=1):
    if train:
        model.train()
    else:
        model.eval()

    imgs = 0
    ips = 0.0
    t_beg = time.perf_counter()

    for index, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # warmup for 2 steps
        if index <= 1:
            t_beg = time.perf_counter()
        else:
            imgs += data.shape[0] * ddp_gpu
            ips = imgs / (time.perf_counter() - t_beg)
            print(progress_bar(imgs, **{'ips': ips}), end="")

        if index + 1 == steps:
            break

    return ips


if __name__ == '__main__':
    config = read_yaml("./example/benchmark/benchmark.yaml")

    if config.get('cudnn', False):
        print('CUDnn enabled.')
        torch.backends.cudnn.benchmark = True
    criterion = getattr(module_loss, config['loss'])
    train = config.get('train', True)
    steps = config.get('steps', None)
    repeat = config.get('repeat', 3)

    use_ddp = config['n_gpu'] > 1 and config.get('ddp', False)
    if use_ddp:
        main_ddp(config)
    else:
        main(config)
