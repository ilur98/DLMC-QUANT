import torch
from numpy import inf
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import MetricTracker
from fnmatch import fnmatch
from trainer.loss import kutosis_loss

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, valid_data_loader=None,
                 lr_scheduler=None, train_log_density=None, valid_log_density=None, rank=-1, world_size=-1):
        super().__init__(model, criterion, metric_ftns, optimizer, config, data_loader, valid_data_loader,
                         lr_scheduler, train_log_density, valid_log_density, rank, world_size)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        if self.config['kurtosis'] is True:
            dict = self.model.state_dict()
            self.conv_weigths = []
            for key in dict.keys():
                #print(key)
                if fnmatch(key, "*conv*weight"):
                    run_key = key[:-6] + 'running_mean'
                    #print(run_key)
                    if run_key not in dict.keys():
                        print(key)
                        self.conv_weigths.append(dict[key])


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        improved = False if self.mnt_mode != 'off' else None
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            if self.config['kurtosis'] is True:
                loss = loss + kutosis_loss(self.conv_weigths, 1.8)
            loss.backward()
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
                    self.train_metrics.reset_batch()
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

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

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
