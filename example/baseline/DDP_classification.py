from logging import log
import sys
sys.path.append(".")

import argparse
import collections
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
from parse_config import ConfigParser
from trainer import Trainer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from logger import NoOp

def mp_deal(fn, config, resume, modification, world_size):
    mp.spawn(fn,
             args=(config, resume, modification, world_size, ),
             nprocs=world_size,
             join=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main(rank, config_parser, resume, modification, world_size):
    config = ConfigParser(config_parser.copy(), resume, modification)
    setup(rank, world_size)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)  
    if dist.get_rank() == 0:
        logger = config.get_logger('Classification', config['trainer']['verbosity'])
        seed = config['random_seed']
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
    else:
        logger = NoOp()
    # setup random seed for random/torch/numpy package

    # setup data_loader instances
    dataloaders = {name: getattr(module_data, cfg['type'])(**cfg['args'])
                   for name, cfg in config['dataloaders'].items()}
    if config['n_gpu'] > 1:
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
        valid_dataloader = torch.utils.data.DataLoader(**valid_args)
    else:
        train_dataloader = dataloaders['train']
        valid_dataloader = dataloaders['test']
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    #print(device)
    #model.to(device)
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    if dist.get_rank() == 0:
        print(logger)
        logger.info(model)
    # print(model)
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', module_lr_scheduler, optimizer, len(train_dataloader))

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=train_dataloader,
                      valid_data_loader=valid_dataloader,
                      lr_scheduler=lr_scheduler,
                      train_log_density=config['trainer'].get('train_log_density', None),
                      valid_log_density=config['trainer'].get('valid_log_density', None),
                      rank=rank,
                      world_size=world_size)

    trainer.train()
    if dist.get_rank() == 0:
        logger.info(f"{trainer.monitor}: {trainer.mnt_best}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Classification')
    args.add_argument('-c', '--config', default="./example/baseline/classification_config.yaml", type=str,
                      help='config file path (default: ./example/baseline/classification_config.yaml)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True 
    config, resume, modification = ConfigParser.from_args(args)
    for ri in range(1):
        mp_deal(main, config, resume, modification, config['n_gpu'])
