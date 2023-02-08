from torch.optim import lr_scheduler
import numpy as np
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