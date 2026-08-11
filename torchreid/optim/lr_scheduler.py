from __future__ import print_function, absolute_import
import torch


AVAI_SCH = ['single_step', 'multi_step', 'cosine', 'warmup_cosine']

# 自定义带线性预热 + 余弦退火调度器
class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epoch, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epoch = max_epoch
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热：从0逐步上升到基础lr
            warmup_ratio = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_ratio for base_lr in self.base_lrs]
        else:
            # 预热结束后执行余弦退火
            cur_epoch = self.last_epoch - self.warmup_epochs
            total_cosine_epoch = self.max_epoch - self.warmup_epochs
            cos_ratio = (1 + torch.cos(torch.tensor(cur_epoch / total_cosine_epoch * torch.pi))) / 2
            return [self.eta_min + (base_lr - self.eta_min) * cos_ratio for base_lr in self.base_lrs]


def build_lr_scheduler(
    optimizer, lr_scheduler='single_step', stepsize=1, gamma=0.1, max_epoch=1,warmup_epochs=5  # 新增预热轮数参数，默认5轮warmup
):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is single_step.
        stepsize (int or list, optional): step size to decay learning rate. When ``lr_scheduler``
            is "single_step", ``stepsize`` should be an integer. When ``lr_scheduler`` is
            "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.

    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
        学习率调度器构建包装函数

        参数说明：
            optimizer (Optimizer): 已经构建完成的优化器对象
            lr_scheduler (str, 可选): 学习率衰减策略，默认 single_step
            stepsize (int / list, 可选): 学习率衰减节点
                - single_step（单步衰减）：stepsize 传整数，每隔固定轮数衰减一次
                - multi_step（多步衰减）：stepsize 传列表，在指定epoch点衰减
                默认值 1
            gamma (float, 可选): 学习率衰减系数，每次衰减 lr = lr * gamma，默认 0.1
            max_epoch (int, 可选): 总训练轮数，余弦退火cosine策略专用，默认 1

        使用示例::
            >>> # 每训练20轮，学习率乘以gamma衰减一次
            >>> scheduler = torchreid.optim.build_lr_scheduler(
            >>>     optimizer, lr_scheduler='single_step', stepsize=20
            >>> )
            >>> # 在第30、50、55轮时分别衰减一次学习率
            >>> scheduler = torchreid.optim.build_lr_scheduler(
            >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
            >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError(
            'Unsupported scheduler: {}. Must be one of {}'.format(
                lr_scheduler, AVAI_SCH
            )
        )

    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

     # 新增带warmup余弦退火分支
    elif lr_scheduler == 'warmup_cosine':
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epoch=max_epoch
        )

    return scheduler
