from fnmatch import fnmatch
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import MetricTracker
import torch.cuda.amp as amp

class QATTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, valid_data_loader=None,
                 lr_scheduler=None, train_log_density=None, valid_log_density=None, rank=-1, world_size=-1):
        super().__init__(model, criterion, metric_ftns, optimizer, config, data_loader, valid_data_loader,
                         lr_scheduler, train_log_density, valid_log_density, rank, world_size)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.update_qparams_period = config['trainer'].get('update_qparams_period', 1)
        self.freeze_bn = config['trainer'].get('freeze_bn', False)
        self.grad_clip_param  = config['grad_clip_param']
    def _freeze_bn(self):
        def _freeze(m):
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        if self.freeze_bn:
            self.model.apply(_freeze)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self._freeze_bn()
        self.train_metrics.reset()
        improved = False if self.mnt_mode != 'off' else None
        #scaler = amp.GradScaler()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            if (epoch * self.len_epoch + batch_idx) % self.update_qparams_period == 1:
                def _reset_qparams(m):
                    if hasattr(m, 'reset_qparams'):
                        m.reset_qparams()
                self.model.apply(_reset_qparams)

            #print(self.device)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            #with amp.autocast():
            output = self.model(data)
            #mean = 0
            #in_upper = 0
            #count = 0
            #for name, param in self.model.named_parameters():
            #    if fnmatch(name, "*in_upper*"):
            #        in_upper = param
            #    elif fnmatch(name, "*in_lower*"):
            #        mean = mean + (in_upper - param)**2
            #        count += 1
            #print(output)
            #loss = self.criterion(output, target) - torch.log(mean/count) + 1
            loss = self.criterion(output, target)
            loss.backward()
            if self.grad_clip_param != 0:
                torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.grad_clip_param)
            #for name, param in self.model.named_parameters():
            #    if param.grad is not None:
            #        print(name)
            #        print(param.mean())
            #        print(param.grad.mean())
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.rank <= 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), len(target))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), len(target))

            if self._time_to_log_train(batch_idx):
                if self.rank <= 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch, self._progress(batch_idx), loss.item())
                    )
                    for k in self.train_metrics.keys:
                        self.writer.add_scalar(k, self.train_metrics.avg_batch(k))
                    for name, param in self.model.named_parameters():
                        if fnmatch(name, "*in_scale*") or fnmatch(name, "*in_lower*") or fnmatch(name, "*wt_alpha*"):
                            self.writer.add_scalar(name, param.data)
                self.train_metrics.reset_batch()
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=Tru
            if self.do_validation and self._time_to_eval(batch_idx) and self.rank <= 0:
                val_log = self._valid_epoch(epoch)

                log = {'epoch': epoch, 'step': batch_idx+1}
                log.update(self.train_metrics.result())
                log.update(**{'val_' + k: v for k, v in val_log.items()})
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                if self.mnt_mode != 'off' and self._is_better(log):
                    self.mnt_best = log[self.mnt_metric]
                    improved = True
                    self._save_best_model(epoch)

        return improved

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step(epoch, 'valid')
                self.valid_metrics.update('loss', loss.item(), len(target))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), len(target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add scalar of metrics to the tensorboard
        for k in self.valid_metrics.keys:
            self.writer.add_scalar(k, self.valid_metrics.avg(k))
        for name, param in self.model.named_parameters():
            if fnmatch(name, "*wt_alpha*"):
                self.writer.add_scalar(name, param.data)
        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

