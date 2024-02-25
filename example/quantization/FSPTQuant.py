from operator import mod
import sys

sys.path.append(".")

import argparse
import random
import copy
import torch
import numpy as np

# from dlmc.quantization.scalar.FSPTQuant.base import FSPTQBase
import data_loader.data_loaders as module_data
import trainer.loss as module_loss
import trainer.metric as module_metric
import model as module_arch
import scheduler.lr_scheduler as module_lr_scheduler
from parse_config import ConfigParser
from trainer import FSPTQTrainer
from fnmatch import fnmatch
from dlmc.utils.quantize import quantize_model
from dlmc.utils.merge_bn import merge_bn
import torch.multiprocessing as mp
from timm.models.resnet import BasicBlock
from model.classification.repvgg import RepVGGBlock, repvgg_model_convert
def get_train_sample(train_loader, num_samples):
    train_datas = []
    for batch in train_loader:
        train_datas.append(batch)
        if len(train_datas) * batch[0].size(0) >= num_samples:
            break
    return train_datas
    # return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]

def main(config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benckmark = True
    torch.backends.cudnn.deterministic = True
    logger = config.get_logger('Quantization aware training')

    seed = config['random_seed']
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    dataloaders = {name: getattr(module_data, cfg['type'])(**cfg['args'])
                   for name, cfg in config['dataloaders'].items()}

    model = config.init_obj('arch', module_arch)

    ckpt_path = config.resume or config['arch'].get('load_from_pth', None)
    if ckpt_path is not None:
        logger.info('Loading checkpoint: {} ...'.format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location='cuda:0')
        if 'state_dict' in checkpoint.keys():
            state_dict = {key.replace("module.", ''): value for key, value in checkpoint['state_dict'].items()}
        else:
            state_dict = {key.replace("module.", ''): value for key, value in checkpoint.items()}
        for name in state_dict.keys():
            if fnmatch(name, "*upper") or fnmatch(name, "*lower"):
                print(state_dict[name].reshape([1]))
                state_dict[name] = state_dict[name].reshape([1])
        model.load_state_dict(state_dict)
    if "RepVGG" in config['arch']['type']:
        model = repvgg_model_convert(model)
    model = merge_bn(model, inplace=True)
    # model.cuda()
    # val, count = 0, 0
    # with torch.no_grad():
    #     for batch_idx, (data, target) in enumerate(dataloaders['test']):
    #         data, target = data.cuda(), target.cuda()
    #         out = model(data)
    #         out = torch.argmax(out, 1)
    #         val = val + (out == target).sum()
    #         count = count + target.numel()

    # print(val / count)
    fp_model = copy.deepcopy(model)
    quantize_model(model, config['quantization'], logger, quantization_type="FSPTQ")
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if fnmatch(name, "*scale*"):
                trainable_params.append(p)

    fs_sampler = get_train_sample(dataloaders['train'], config['train_sample_num'])
    model.cuda()
    model.eval()

    print("initial_done.\n")
    trainer = FSPTQTrainer(model, fp_model, criterion, metrics, optimizer_dict=config['optimizer']['args'],
                        config=config,
                        data_loader=fs_sampler,
                        valid_data_loader=dataloaders['test'],
                        block_dict=[BasicBlock, RepVGGBlock],
                        lr_scheduler=None,
                        train_log_density=config['trainer'].get('train_log_density', None),
                        valid_log_density=config['trainer'].get('valid_log_density', None),
                        world_size=1)
    trainer.train()
    logger.info(f"{trainer.monitor}: {trainer.mnt_best}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Quantization aware training')
    args.add_argument('-c', '--config', default='./example/quantization/FSPTQ_config.yaml', type=str,
                      help='config file path (default: ./example/quantization/FSPTQ_config.yaml')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config, resume, modification = ConfigParser.from_args(args)
    for i in range(1):
        config_parser = ConfigParser(config.copy(), resume, modification)
        main(config_parser)
    