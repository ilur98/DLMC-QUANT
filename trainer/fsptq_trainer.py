from cgitb import handler
from dataclasses import dataclass
from multiprocessing.dummy import active_children
from pyexpat import model
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from dlmc.quantization.scalar.FSPTQuant.base import FSPTQBase
from utils import MetricTracker
from functools import partial
from trainer.loss import cross_entropy_loss
class FSPTQTrainer(BaseTrainer):

    def __init__(self, model, fp_model, criterion, metric_ftns, optimizer_dict, config, data_loader, valid_data_loader=None,
                 lr_scheduler=None, block_dict=None, train_log_density=None, valid_log_density=None, rank=-1, world_size=-1):
        super().__init__(model, criterion, metric_ftns, None, config, data_loader, valid_data_loader, 
                            lr_scheduler, train_log_density, valid_log_density, rank, world_size)
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker(
            'loss',
            *[m.__name__ for m in self.metric_ftns]
        )
        self.block_dict = block_dict
        self.optimizer_dict = optimizer_dict
        self.fp_model = fp_model

    def train(self):
        self.train_metrics.reset()
        # val_log = self._valid_epoch(1)

        # log = {'epoch': 0, 'step': 50}
        # # log.update(self.train_metrics.result())
        # log.update(**{'val_' + k: v for k, v in val_log.items()})
        # for key, value in log.items():
        #     self.logger.info('    {:15s}: {}'.format(str(key), value))
        def Rhook(name, module, input, output):
            # cached_input.append(input[0].cpu())
            cached_output.append(output.cpu())
        
        def Rhook_quantize(name, module, input, output):
            cached_input.append(input[0].cpu())

        for (name, module), fp_module in zip(self.model.named_modules(), self.fp_model.modules()):
            cached_input, cached_output = [], []
            cached_quant = []
            handler_list = []
            # print(name, module, fp_module)
            # print(type(module))
            if isinstance(module, FSPTQBase) and name in ["conv1", "linear"]:
                print('Reconstruction for layer {}'.format(name))
                handler_list.append(fp_module.register_forward_hook(partial(Rhook, name)))
                handler_list.append(module.register_forward_hook(partial(Rhook_quantize, name)))
            elif type(module) in self.block_dict:
                print('Reconstruction for block {}'.format(name))
                handler_list.append(fp_module.register_forward_hook(partial(Rhook, name)))
                handler_list.append(module.register_forward_hook(partial(Rhook_quantize, name)))
            else:
                continue
            self.model.cuda().eval()
            self.fp_model.cuda().eval()
            # print(self.device)
            for batch_idx, (data, target) in enumerate(self.data_loader):
                with torch.no_grad():
                    data = data.cuda()
                    _ = self.fp_model(data)
                    _ = self.model(data)
            block_input = torch.cat(cached_input)
            block_output = torch.cat(cached_output)
            torch.cuda.empty_cache()
            block_input = block_input.to(self.device)
            block_output = block_output.to(self.device)
            for handler in handler_list:
                handler.remove()
            self.fp_model.cpu()
            optimizer, scheduler = self.generate_optimizer(module)
            self.model.train()
            for i in range(self.epochs):
                idx = torch.randperm(block_input.size(0))[:64]
                optimizer.zero_grad()
                cur_input = block_input[idx]
                if i == 0:
                    print(cur_input.max(), cur_input.min())
                # cur_input = torch.where((torch.rand_like(cur_input) > 0.5), cur_input, block_quant[idx])
                cur_output = module(cur_input)
                loss = self.criterion(block_output[idx], cur_output)
                # round_loss = 0
                # if i < 0.2 * 20000:
                #     round_loss = torch.tensor(0.)
                # else:
                #     b = (2 - 20) * (i - 0.2 * 20000) / (20000 - 0.2 * 20000) + 20
                #     for m in module.modules():
                #         if isinstance(m, FSPTQBase):
                #             round_vals = m.get_soft_targets()
                #             round_loss += 0.01 * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                #     loss += round_loss
                loss.backward(retain_graph=True)
                # print(m.alpha.grad)
                optimizer.step()
                scheduler.step()
                if i % 500 == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        i, self._progress(batch_idx), loss.item()))
        print("start val")
        val_log = self._valid_epoch(1)

        log = {'epoch': 0, 'step': batch_idx+1}
        # log.update(self.train_metrics.result())
        log.update(**{'val_' + k: v for k, v in val_log.items()})
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
        self._save_best_model(0)

    def _valid_epoch(self, epoch):
        self.model.cuda().eval()
        self.fp_model.cuda().eval()
        self.valid_metrics.reset()
        self.change_model_state(self.model, True, True)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output_fp = self.fp_model(data)
                output = self.model(data)
                loss = self.criterion(output_fp, output)
                print("l2loss", loss)
                loss = cross_entropy_loss(output, target)
                self.writer.set_step(epoch, 'valid')
                self.valid_metrics.update('loss', loss.item(), len(target))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), len(target))
                
        return self.valid_metrics.result()
                
                

    def generate_optimizer(self, module):
        opt_dict = []
        for name, param in module.named_parameters():
            if name.endswith("weight"):
                opt_dict.append({'params': param, 'lr' : 1e-5})
            elif name.endswith("scales"):
                opt_dict.append({'params': param, 'lr' : 1e-3})
            elif name.endswith("bias"):
                opt_dict.append({'params':param, 'lr': 1e-5})
            elif name.endswith("gamma") or name.endswith("beta"):
                opt_dict.append({"params": param, 'lr' : 0.1})
            else:
                opt_dict.append({"params": param, 'lr': 1e-5})
        optimizer = torch.optim.Adam(opt_dict)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0.)

        return optimizer, scheduler


    def change_model_state(self, module, wt_state, act_state):
        for name, m in module.named_modules():
            if isinstance(m, FSPTQBase):
                if name == "conv1":
                    m.change_quant_state(wt_state, False)
                else:
                    m.change_quant_state(wt_state, act_state)
