import sys
sys.path.append(".")

import argparse
import collections
import random

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


def main(config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True
    logger = config.get_logger('Classification')

    # setup random seed for random/torch/numpy package
    seed = config['random_seed']
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # setup data_loader instances
    dataloaders = {name: getattr(module_data, cfg['type'])(**cfg['args'])
                   for name, cfg in config['dataloaders'].items()}

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    # logger.info(model)
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', module_lr_scheduler, optimizer, len(dataloaders['train']))

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=dataloaders['train'],
                      valid_data_loader=dataloaders.get('test', None),
                      lr_scheduler=lr_scheduler,
                      train_log_density=config['trainer'].get('train_log_density', None),
                      valid_log_density=config['trainer'].get('valid_log_density', None),
                      world_size=1)

    trainer.train()
    logger.info(f"{trainer.monitor}: {trainer.mnt_best}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Classification')
    args.add_argument('-c', '--config', default="./example/baseline/classification_config.yaml", type=str,
                      help='config file path (default: ./example/baseline/classification_config.yaml)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config, resume, modification = ConfigParser.from_args(args)
    for ri in range(3):
        config_parser = ConfigParser(config.copy(), resume, modification)
        main(config_parser)
