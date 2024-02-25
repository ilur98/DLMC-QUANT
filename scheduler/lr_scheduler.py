import math
import numpy as np
from abc import abstractmethod
from bisect import bisect_right
from torch.optim.optimizer import Optimizer


class _LRScheduler(object):

    def __init__(self, optimizer, steps_per_epc, cur_steps=0, warmup_steps=0, world_size=1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize step counters and base learning rates
        cur_steps = max(0, cur_steps)
        if cur_steps == 0:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.steps = cur_steps
        self._steps_per_epc = steps_per_epc
        self._warmup_steps = max(0, warmup_steps)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        self.steps = state_dict['steps']

    @abstractmethod
    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.steps += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class MultiStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps_per_epc (int): Steps count per training epoch.
        milestones (list): List of epoch indices when learning rates
            decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        cur_steps (int): The index of current steps, usually used in
            fine-tuning. Default: 0.
        warmup_steps (int): The Steps count for linear warm-up at the
            beginning of training. Default: 0.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.~0.05  if steps < 100
        >>> # lr = 0.05     if steps >= 100 and epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1, warmup_steps=100)
        >>> for epoch in range(100):
        >>>     for (data, target) in data_loader:
        >>>         ...
        >>>         optimizer.step()
        >>>         scheduler.step()
    """

    def __init__(self, optimizer, steps_per_epc, milestones, gamma=0.1, cur_steps=0, warmup_steps=0):
        if not isinstance(milestones, list):
            raise TypeError('`reduce_steps` must be a `list` representing the steps when lr should be reduced')
        if gamma <= 0.0 or gamma >= 1.0:
            raise ValueError('`gamma` should be between (0.0, 1.0), but get {}.'.format(str(gamma)))

        self.milestones = sorted([_ * steps_per_epc for _ in milestones])
        self.gamma = gamma

        super(MultiStepLR, self).__init__(optimizer, steps_per_epc, cur_steps, warmup_steps)

    def get_lr(self):
        if self.steps < self._warmup_steps:
            mult = self.steps / self._warmup_steps
        else:
            mult = self.gamma ** bisect_right(self.milestones, self.steps)
        return [base_lr * mult for base_lr in self.base_lrs]


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps_per_epc (int): Steps count per training epoch.
        gamma (float): Factor by which the learning rate will be
            reduced. new_lr = lr * gamma. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr_mult (float or list): A scalar or a list of scalars.
            A lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        cur_steps (int): The index of current steps, usually used in
            fine-tuning. Default: 0.
        warmup_steps (int): The Steps count for linear warm-up at the
            beginning of training. Default: 0.

    Example:
        >>> scheduler = ReduceLROnPlateau(optimizer, steps_per_epc)
        >>> for epoch in range(100):
        >>>     for (data, target) in data_loader:
        >>>         ...
        >>>         optimizer.step()
        >>>         scheduler.step(metrics)
    """

    def __init__(self, optimizer, steps_per_epc, gamma=0.1, patience=10, mode='min', threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr_mult=0, cur_steps=0, warmup_steps=0):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize step counters and base learning rates
        cur_steps = max(0, cur_steps)
        if cur_steps == 0:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

        # Initialize ``gamma`` for each learning rate decay
        if gamma <= 0.0 or gamma >= 1.0:
            raise ValueError('`gamma` should be between (0.0, 1.0), but get {}.'.format(str(gamma)))
        self.gamma = gamma

        # Initialize steps counters
        self._steps_per_epc = steps_per_epc
        self._warmup_steps = max(0, warmup_steps)
        self._steps_in_epc = 0
        self._metrics_in_epc = []
        self.steps = cur_steps

        if isinstance(min_lr_mult, list) or isinstance(min_lr_mult, tuple):
            if len(min_lr_mult) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr_mult)))
            self.min_lr_mults = list(min_lr_mult)
        else:
            self.min_lr_mults = [min_lr_mult] * len(optimizer.param_groups)

        # Initialize plateau definition config
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = cooldown
        self.num_bad_epochs = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.eps = 1e-8
        self.cur_lr_mult = 1.0
        self.mode_worse = None  # the worse value for the chosen mode

        self._init_is_better(mode, threshold, threshold_mode)
        self.best = self.mode_worse

    def get_lr(self):
        if self.steps < self._warmup_steps:
            mult = self.steps / self._warmup_steps
            return [base_lr * mult for base_lr in self.base_lrs]
        else:
            mults = [max(self.cur_lr_mult, min_lr) for min_lr in self.min_lr_mults]
            return [base_lr * mult for base_lr, mult in zip(self.base_lrs, mults)]

    def step(self, metrics):
        """
        This is the outer-field of ``step``, since lr_scheduler.step()
        is called each step but the updating of learning rate is based
        on epoch. This outer-field only acts as a step counter, and
        call the inner-field ``_step`` at the end of each epoch.

        :param metrics: The observed metrics
        """
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.steps += 1
        self._steps_in_epc += 1
        self._metrics_in_epc.append(current)

        if self._steps_in_epc == self._steps_per_epc:
            self._step(np.mean(self._metrics_in_epc))
            self._steps_in_epc = 0
            self._metrics_in_epc.clear()

    def _step(self, metrics):
        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.cur_lr_mult *= self.gamma
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        import math

        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        self.mode_worse = math.inf if mode == 'min' else -math.inf
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state, init metrics comparing method.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)


class CosineCyclicLR(_LRScheduler):
    """Cosine cyclic learning rate scheduler

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps_per_epc (int): Steps count per training epoch.
        Tepoch (int): Number of epochs of half a cosine cycle.
            Default: 5
        cycles (int): Number of cosine cycles. Since models are best
            when learning rate is minimal, which is at half of a
            cosine cycle, so the number of real cycles would be
            ``cycles`` + `0.5` instead of ``cycles``.
            Default: 10
        min_lr_mult (float): The minimal learning rate mult, which
            occurs at the half of each cosine cycles.
        cur_steps (int): The index of current steps, usually used in
            fine-tuning. Default: 0.
        warmup_steps (int): The Steps count for linear warm-up at the
            beginning of training. Default: 0.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # each epoch contains 500 steps and the training
        >>> # process includes 100 epochs in total.
        >>> # lr = 0.~0.05    if steps < 100
        >>> # lr = 0.02525 + 0.02475*cos(1/(5*500) * steps * pi)
                              otherwise
        >>> scheduler = CosineCyclicLR(optimizer, steps_per_epc, Tepoch=5,
        cycles=10, min_lr_mult=0.01, warmup_steps=100)
        >>> for epoch in range(100):
        >>>     for (data, target) in data_loader:
        >>>         ...
        >>>         optimizer.step()
        >>>         scheduler.step()
    """

    def __init__(self, optimizer, steps_per_epc, Tepoch=5, cycles=10, min_lr_mult=0., cur_steps=0, warmup_steps=0):
        self.Tsteps = Tepoch * steps_per_epc
        self.cycles = cycles
        self.min_lr_mult = min_lr_mult
        if min_lr_mult < 0. or min_lr_mult >= 1.:
            raise ValueError('`min_lr_mult` must between [0.0, 1.0), but get {}.'.format(min_lr_mult))

        super(CosineCyclicLR, self).__init__(optimizer, steps_per_epc, cur_steps, warmup_steps)

    def get_lr(self):
        if self.steps < self._warmup_steps:
            mult = self.steps / self._warmup_steps
        elif self.steps < self._warmup_steps + self.Tsteps * (self.cycles * 2 + 1):
            scale = (1. - self.min_lr_mult) / 2.
            bias = (1. + self.min_lr_mult) / 2.
            r = self.steps / self.Tsteps
            mult = scale * math.cos(r * math.pi) + bias
        else:
            mult = self.min_lr_mult
        return [base_lr * mult for base_lr in self.base_lrs]


class CosineDecayLR(CosineCyclicLR):
    """Decays the learning rate of each parameter group using cosine
    decay, learning rate gradually decreases from the initial learning
    rate to `0`, corresponding to the [0, pi] field of cosine function.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps_per_epc (int): Steps count per training epoch.
        total_epochs (int): Number of epochs of training.
        cur_steps (int): The index of current steps, usually used in
            fine-tuning. Default: 0.
        warmup_steps (int): The Steps count for linear warm-up at the
            beginning of training. Default: 0.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups,
        >>> # each epoch contains 500 steps and the training
        >>> # process includes 100 epochs in total.
        >>> # lr = 0.~0.05                                   if steps < 100
        >>> # lr = 0.025 + 0.025*cos(1/49900 * steps * pi)   otherwise
        >>> scheduler = CosineDecayLR(optimizer, steps_per_epc, warmup_steps=100)
        >>> for epoch in range(100):
        >>>     for (data, target) in data_loader:
        >>>         ...
        >>>         optimizer.step()
        >>>         scheduler.step()
    """

    def __init__(self, optimizer, steps_per_epc, total_epochs, cur_steps=0, warmup_steps=0):
        super(CosineDecayLR, self).__init__(
            optimizer, steps_per_epc, Tepoch=total_epochs, cycles=0,
            min_lr_mult=0., cur_steps=cur_steps, warmup_steps=warmup_steps
        )


class CosineAnnealingLR(_LRScheduler):
    """Cosine annealing learning rate scheduler

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps_per_epc (int): Steps count per training epoch.
        Tepoch (int): Number of epochs of a cosine annealing cycle,
            which correspond to half a cosine cycle.
            Default: 10
        cycles (int): Number of cosine annealing cycles.
            Default: 10
        min_lr_mult (float): The minimal learning rate mult, which
            occurs at the end of each cosine annealing cycles.
        cur_steps (int): The index of current steps, usually used in
            fine-tuning. Default: 0.
        warmup_steps (int): The Steps count for linear warm-up at the
            beginning of training. Default: 0.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # each epoch contains 500 steps and the training
        >>> # process includes 100 epochs in total.
        >>> # lr = 0.~0.05    if steps < 100
        >>> # lr = 0.02525 + 0.02475*cos(1/(10*500) * (steps%(10*500) * pi)
                              otherwise
        >>> scheduler = CosineAnnealingLR(optimizer, steps_per_epc, Tepoch=10,
        cycles=10, min_lr_mult=0.01, warmup_steps=100)
        >>> for epoch in range(100):
        >>>     for (data, target) in data_loader:
        >>>         ...
        >>>         optimizer.step()
        >>>         scheduler.step()
    """

    def __init__(self, optimizer, steps_per_epc, Tepoch=10, cycles=10, min_lr_mult=0., cur_steps=0, warmup_steps=0):
        self.Tsteps = Tepoch * steps_per_epc
        self.cycles = cycles
        self.min_lr_mult = min_lr_mult
        if min_lr_mult < 0. or min_lr_mult >= 1.:
            raise ValueError('`min_lr_mult` must between [0.0, 1.0), but get {}.'.format(min_lr_mult))

        super(CosineAnnealingLR, self).__init__(optimizer, steps_per_epc, cur_steps, warmup_steps)

    def get_lr(self):
        if self.steps < self._warmup_steps:
            mult = self.steps / self._warmup_steps
        elif self.steps < self._warmup_steps + self.Tsteps * self.cycles:
            scale = (1. - self.min_lr_mult) / 2.
            bias = (1. + self.min_lr_mult) / 2.
            r = (self.steps % self.Tsteps) / self.Tsteps
            mult = scale * math.cos(r * math.pi) + bias
        else:
            mult = self.min_lr_mult
        return [base_lr * mult for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps_per_epc (int): Steps count per training epoch.
        gamma (float): Multiplicative factor of learning rate decay.
        cur_steps (int): The index of current steps, usually used in
            fine-tuning. Default: 0.
        warmup_steps (int): The Steps count for linear warm-up at the
            beginning of training. Default: 0.
    """

    def __init__(self, optimizer, steps_per_epc, gamma, cur_steps=0, warmup_steps=0):
        self.gamma = gamma ** (1.0 / steps_per_epc)
        super(ExponentialLR, self).__init__(optimizer, steps_per_epc, cur_steps, warmup_steps)

    def get_lr(self):
        if self.steps < self._warmup_steps:
            mult = self.steps / self._warmup_steps
        else:
            mult = self.gamma ** (self.steps - self._warmup_steps)
        return [base_lr * mult for base_lr in self.base_lrs]
