import sys
sys.path.append(".")

import argparse
import random
from tqdm import tqdm

import torch
import numpy as np

import sys
sys.path.append(".")
import data_loader.data_loaders as module_data
import trainer.loss as module_loss
import trainer.metric as module_metric
import model as module_arch
from parse_config import ConfigParser

from dlmc.utils.quantize import quantize_model


def main(config):
    logger = config.get_logger('Post training quantization')

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

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load checkpoint
    if config.resume is not None:
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = {key.replace("module.", ''): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

    # apply quantization
    quantize_model(model, config['quantization'], logger)

    # wrap the model with DataParallel
    # TODO: replace DataParallel with distributed
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # do calibration
    with torch.no_grad():
        for data, _ in tqdm(dataloaders['calibration'], desc="Calib"):
            data = data.to(device)
            _ = model(data)

    # evaluate
    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(dataloaders['test'], desc="Eval")):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    # print results
    n_samples = len(dataloaders['test'].sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    # save quantized model
    state = {
        'state_dict': model.state_dict(),
        'config': config,
        **log
    }
    filename = str(config.save_dir / "quantized_model.pth")
    torch.save(state, filename)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Post-training Quantization')
    args.add_argument('-c', '--config', default='./example/quantization/PTQ_config.yaml', type=str,
                      help='config file path (default: ./example/quantization/PTQ_config.yaml')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config, resume, modification = ConfigParser.from_args(args)
    config_parser = ConfigParser(config.copy(), resume, modification)

    main(config_parser)
