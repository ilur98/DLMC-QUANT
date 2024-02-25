import sys
from threading import local

from torch.nn.parallel import distributed

sys.path.append(".")

import argparse
import random
import os
import torch
import numpy as np

import sys
sys.path.append(".")
import data_loader.data_loaders as module_data
import trainer.loss as module_loss
import trainer.metric as module_metric
import model as module_arch
import scheduler.lr_scheduler as module_lr_scheduler
import torch.distributed as dist
from parse_config import ConfigParser
from trainer import QATTrainer
from fnmatch import fnmatch
from dlmc.utils.quantize import quantize_model
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from logger import NoOp

def mp_deal(fn, config, resume, modification, world_size):
    mp.spawn(fn,
             args=(config, resume, modification, world_size, ),
             nprocs=world_size,
             join=True)

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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main(rank, config_parser, resume, modification, world_size):
    config = ConfigParser(config_parser.copy(), resume, modification)
    setup(rank, world_size)
    local_rank = torch.distributed.get_rank()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)    
    if dist.get_rank() == 0:
        logger = config.get_logger('Quantization aware training')
        print(type(logger))
    else:
        logger = NoOp()
    # setup random seed for random/torch/numpy package
    seed = config['random_seed']
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # setup data_loader instances
    dataloaders = {name: getattr(module_data, cfg['type'])(**cfg['args'])
                   for name, cfg in config['dataloaders'].items()}
    train_ddp_sampler = DistributedSampler(dataloaders['train'].dataset)
    valid_ddp_sampler = DistributedSampler(dataloaders['test'].dataset)
    train_args = {
        'dataset': dataloaders['train'].dataset,
        'batch_size': dataloaders['train'].batch_size,
        'sampler': train_ddp_sampler,
        'num_workers': dataloaders['train'].num_workers,
        'drop_last': dataloaders['train'].drop_last
    }
    valid_args = {
        'dataset': dataloaders['test'].dataset,
        'batch_size': dataloaders['test'].batch_size,
        'sampler': valid_ddp_sampler,
        'num_workers': dataloaders['test'].num_workers,
        'drop_last': dataloaders['test'].drop_last
    }
    train_dataloader = torch.utils.data.DataLoader(**train_args)
    #valid_dataloader = torch.utils.data.DataLoader(**valid_args)
    # build model architecture
    model = config.init_obj('arch', module_arch)

    if dist.get_rank() == 0:
        print(model)
        logger.info(model)

    # load checkpoint
    if dist.get_rank() == 0:
        ckpt_path = config.resume or config['arch'].get('load_from_pth', None)
        if ckpt_path is not None:
            logger.info('Loading checkpoint: {} ...'.format(ckpt_path))
            checkpoint = torch.load(ckpt_path) 
            state_dict = {key.replace("module.", ''): value for key, value in checkpoint['state_dict'].items()}
            for name in state_dict.keys():
                if fnmatch(name, "*upper") or fnmatch(name, "*lower"):
                    print(state_dict[name].reshape([1]))
                    state_dict[name] = state_dict[name].reshape([1])
            model.load_state_dict(state_dict)
    quantize_model(model, config['quantization'], logger, quantization_type="RootQ")
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    print(model.device)
    # apply quantization
    

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = []
    alpha_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if fnmatch(name, "*wt_alpha*"):
                alpha_params.append(p)
            else:
                trainable_params.append(p)
    params = [{'params': trainable_params},
              {'params': alpha_params, 'lr': config['alpha_lr']}]
    #for param in model.parameters():  
    #    print(type(param), param.size())
    optimizer = config.init_obj('optimizer', torch.optim, params)

    lr_scheduler = config.init_obj('lr_scheduler', module_lr_scheduler, optimizer, len(train_dataloader))

    trainer = QATTrainer(model, criterion, metrics, optimizer,
                         config=config,
                         data_loader=train_dataloader,
                         valid_data_loader=dataloaders['test'],
                         lr_scheduler=lr_scheduler,
                         train_log_density=config['trainer'].get('train_log_density', None),
                         valid_log_density=config['trainer'].get('valid_log_density', None),
                         rank=dist.get_rank(),
                         world_size=dist.get_world_size())

    trainer.train()
    if dist.get_rank() == 0:
        logger.info(f"{trainer.monitor}: {trainer.mnt_best}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Quantization aware training')
    args.add_argument('-c', '--config', default='./example/quantization/RootQ_config.yaml', type=str,
                      help='config file path (default: ./example/quantization/RootQ_config.yaml')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--local_rank', default=None)
    config, resume, modification = ConfigParser.from_args(args)
    for i in range(1):
        mp_deal(main, config, resume, modification, config['n_gpu'])
