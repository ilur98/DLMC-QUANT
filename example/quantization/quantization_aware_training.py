import sys
sys.path.append(".")

import argparse
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
from trainer import QATTrainer

from dlmc.utils.quantize import quantize_model


def main(config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    logger = config.get_logger('Quantization aware training')

    # setup random seed for random/torch/numpy package
    seed = config['random_seed']
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # setup data_loader instances
    dataloaders = {name: getattr(module_data, cfg['type'])(**cfg['args'])
                   for name, cfg in config['dataloaders'].items()}

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # load checkpoint
    ckpt_path = config.resume or config['arch'].get('load_from_pth', None)
    if ckpt_path is not None:
        logger.info('Loading checkpoint: {} ...'.format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        state_dict = {key.replace("module.", ''): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    model.to(torch.device('cuda'))
    model.eval()
    sum1 = 0
    sum2 = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloaders['test']):
            data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
            output = model(data)
            sum1 = sum1 + (output.argmax(axis=1)==target).sum()
            sum2 = sum2 + target.numel()
    print(sum1/sum2)

    # apply quantization
    quantize_model(model, config['quantization'], logger)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    #for param in model.parameters():  
    #    print(type(param), param.size())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', module_lr_scheduler, optimizer)

    trainer = QATTrainer(model, criterion, metrics, optimizer,
                         config=config,
                         data_loader=dataloaders['train'],
                         valid_data_loader=dataloaders['test'],
                         lr_scheduler=lr_scheduler,
                         train_log_density=config['trainer'].get('train_log_density', None),
                         valid_log_density=config['trainer'].get('valid_log_density', None))

    trainer.train()
    logger.info(f"{trainer.monitor}: {trainer.mnt_best}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Quantization aware training')
    args.add_argument('-c', '--config', default='./example/quantization/QAT_config.yaml', type=str,
                      help='config file path (default: ./example/quantization/QAT_config.yaml')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config, resume, modification = ConfigParser.from_args(args)
    config_parser = ConfigParser(config.copy(), resume, modification)

    main(config_parser)
