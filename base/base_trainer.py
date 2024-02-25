import torch
import random
import numpy as np
import os
import torch.distributed as dist
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from fnmatch import fnmatch
class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, valid_data_loader,
                 lr_scheduler, train_log_density, valid_log_density, rank=-1, world_size=-1):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        # setup GPU device if available, move model into configured device
        # TODO: replace DataParallel with distributed
        self.device, device_ids = self._prepare_device(world_size, rank)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        self.rank = rank
        self.world_size = world_size
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.len_epoch = len(self.data_loader)
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        train_log_density = int(np.ceil(train_log_density) or np.sqrt(self.len_epoch))
        valid_log_density = int(np.ceil(valid_log_density) or 1)
        self.train_log_step = [np.round(self.len_epoch * idx / train_log_density)
                               for idx in range(1, train_log_density+1)]
        self.valid_log_step = [np.round(self.len_epoch * idx / valid_log_density)
                               for idx in range(1, valid_log_density+1)]

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer.get('save_period', 10)
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.save_to_disk = cfg_trainer.get('save_to_disk', True)

        self.last_best_path = None

        self.rd_seed_per_epoch = np.random.randint(
            low=-0x7fffffffffffffff, high=0x7fffffffffffffff, size=self.epochs+1, dtype=np.int64)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = max(1, cfg_trainer.get('early_stop', inf))

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        if self.rank <= 0: 
            self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer.get('tensorboard', False))

        if config.resume is None:
            self._save_checkpoint(0)
        else:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Fixed the random seed per epoch for the convenience of reproducible fine-tuning.
            # Each round of dataloader shuffle, or dropout, dropblock can either influence the
            # random number state.
            manual_seed(self.rd_seed_per_epoch[epoch])

            improved = self._train_epoch(epoch)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            if self.mnt_mode != 'off':
                if improved:
                    not_improved_count = 0
                else:
                    not_improved_count += 1
                    if not_improved_count > self.early_stop:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.early_stop))
                        break
            

    def _prepare_device(self, n_gpu_use, rank):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        print(n_gpu_use)
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        if rank == -1:
            device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        else:
            device = torch.device("cuda", rank)
        list_ids = list(range(n_gpu_use))
        #print(device, list_ids)
        return device, list_ids

    def _save_best_model(self, epoch):
        """
        Saving best model during training

        :param epoch: current epoch number
        """
        if not self.save_to_disk:
            return

        state = {'epoch': epoch}
        state.update(self.state)
        if self.mnt_mode == 'off':
            file_name = 'model_best.pth'
        else:
            file_name = str('model_best-{}.pth'.format(str(self.mnt_best)))
        best_path = str(self.checkpoint_dir / file_name)
        torch.save(state, best_path)

        # Delete the obsolete `best_model`
        if self.last_best_path is not None:
            try:
                os.remove(self.last_best_path)
            except OSError:
                self.logger.warning('Deleting non-exist file ' + self.last_best_path)
        self.last_best_path = best_path

        self.logger.info("Saving current best: {} ...".format(best_path))

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        if not self.save_to_disk:
            return

        state = {'epoch': epoch}
        state.update(self.state)
        if self.mnt_mode == 'off':
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}-{}.pth'.format(epoch, self.mnt_best))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # To ensure reproducible fine-tuning, we must ensure the random number state is same as original
        # at any time, which enforce the current training process to have the same `valid_log_density`
        # as original.
        if checkpoint['config'].config.get('random_seed', None) is not None:
            orig_valid_log_density = checkpoint['config']['trainer'].get('valid_log_density', 1)
            cur_valid_log_density = self.config['trainer'].get('valid_log_density', 1)
            if orig_valid_log_density != cur_valid_log_density:
                self.logger.warning("Warning: For reproducible fine-tuning, the current `valid_log_density`({}) must "
                                    "match the original `valid_log_density`({}). This fine-tuning may get different"
                                    "result from original".format(orig_valid_log_density, cur_valid_log_density))

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        for name, param in checkpoint['state_dict'].items():
            if fnmatch(name, "*upper") or fnmatch(name, "*lower"):
                checkpoint['state_dict'][name] = param.reshape([1])
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # load learning rate scheduler state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['lr_scheduler']['type'] != self.config['lr_scheduler']['type']:
            self.logger.warning("Warning: Learning rate scheduler type given in config file is different from that "
                                "of checkpoint. Learning rate scheduler parameters not being resumed.")
        else:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _time_to_log_train(self, batch_idx):
        return (batch_idx + 1) in self.train_log_step

    def _time_to_eval(self, batch_idx):
        return (batch_idx + 1) in self.valid_log_step

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = (batch_idx+1) * self.data_loader.batch_size
            total = self.data_loader.n_samples
            current = min(current, total)
        else:
            current = batch_idx+1
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _is_better(self, log):
        assert self.mnt_mode != 'off'
        improved = False
        try:
            # check whether model performance improved or not, according to specified metric(mnt_metric)
            improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
        except KeyError:
            self.logger.warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(self.mnt_metric))
            self.mnt_mode = 'off'
            self.mnt_best = 0
        return improved

    @property
    def state(self):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'state_dict': self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state

def manual_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(np.uint32(seed >> 32))      # numpy.random accept 32-bit seed
