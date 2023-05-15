from torch.optim import lr_scheduler
import numpy as np
import torch 

class CosineWarmupLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warmup=0, T_max=10):
        """ Description: get warmup cosine lr scheduler
        :param optimizer: (torch.optim.*), torch optimizer
        :param lr_min: (float), minimum learning rate
        :param lr_max: (float), maximum learning rate
        :param warmup: (int), warm up iterations
        :param T_max: (int), maximum number of steps
        Example:
        <<< epochs = 100
        <<< warm_up = 5
        <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
        <<< lrs = []
        <<< for epoch in range(epochs):
        <<< optimizer.step()
        <<< lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        <<< cosine_lr.step()
        <<< plt.plot(lrs, color='r')
        <<< plt.show() """

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup = warmup
        self.T_max = T_max
        self.cur = 0

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warmup == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warmup != 0) & (self.cur <= self.warmup):
            lr = self.lr_min + (self.lr_max - self.lr_min) * self.cur / self.warmup
        else:
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
                 (np.cos((self.cur - self.warmup) / (self.T_max - self.warmup) * np.pi) + 1)

        self.cur += 1

        return [lr for _ in self.base_lrs]
    

class LinearLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1, verbose=False):
        """Decays the learning rate of each parameter group by linearly changing small
        multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
        Notice that such decay can happen simultaneously with other changes to the learning rate
        from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            start_factor (float): The number we multiply learning rate in the first epoch.
                The multiplication factor changes towards end_factor in the following epochs.
                Default: 1./3.
            end_factor (float): The number we multiply learning rate at the end of linear changing
                process. Default: 1.0.
            total_iters (int): The number of iterations that multiplicative factor reaches to 1.
                Default: 5.
            last_epoch (int): The index of the last epoch. Default: -1.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.

        Example:
            # xdoctest: +SKIP
            # Assuming optimizer uses lr = 0.05 for all groups
            # lr = 0.025    if epoch == 0
            # lr = 0.03125  if epoch == 1
            # lr = 0.0375   if epoch == 2
            # lr = 0.04375  if epoch == 3
            # lr = 0.05    if epoch >= 4
            scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
            for epoch in range(100):
                train(...)
                validate(...)
                scheduler.step()
        """

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.lr = optimizer.param_groups[0]['lr']
        super().__init__(optimizer, -1)

    def get_lr(self):
        if self.last_epoch == 0:
            self.lr = self.lr * self.start_factor
            return [self.lr for _ in self.base_lrs]

        if self.last_epoch > self.total_iters:
            self.lr = self.lr
            return [self.lr for _ in self.base_lrs]

        self.lr = self.lr * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
        return [self.lr for _ in self.base_lrs]