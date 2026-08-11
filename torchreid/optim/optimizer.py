from __future__ import print_function, absolute_import
import warnings
import torch
import torch.nn as nn

from .radam import RAdam

AVAI_OPTIMS = ['adam', 'amsgrad', 'sgd', 'rmsprop', 'radam', 'adamw']


def build_optimizer(
    model,
    optim='adam',
    lr=0.0003,
    weight_decay=5e-04,
    momentum=0.9,
    sgd_dampening=0,
    sgd_nesterov=False,
    rmsprop_alpha=0.99,
    adam_beta1=0.9,
    adam_beta2=0.99,
    staged_lr=False,
    new_layers='',
    base_lr_mult=0.1
):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module): model.
        optim (str, optional): optimizer. Default is "adam".
        lr (float, optional): learning rate. Default is 0.0003.
        weight_decay (float, optional): weight decay (L2 penalty). Default is 5e-04.
        momentum (float, optional): momentum factor in sgd. Default is 0.9,
        sgd_dampening (float, optional): dampening for momentum. Default is 0.
        sgd_nesterov (bool, optional): enables Nesterov momentum. Default is False.
        rmsprop_alpha (float, optional): smoothing constant for rmsprop. Default is 0.99.
        adam_beta1 (float, optional): beta-1 value in adam. Default is 0.9.
        adam_beta2 (float, optional): beta-2 value in adam. Default is 0.99,
        staged_lr (bool, optional): uses different learning rates for base and new layers. Base
            layers are pretrained layers while new layers are randomly initialized, e.g. the
            identity classification layer. Enabling ``staged_lr`` can allow the base layers to
            be trained with a smaller learning rate determined by ``base_lr_mult``, while the new
            layers will take the ``lr``. Default is False.
        new_layers (str or list): attribute names in ``model``. Default is empty.
        base_lr_mult (float, optional): learning rate multiplier for base layers. Default is 0.1.

    Examples::
        >>> # A normal optimizer can be built by
        >>> optimizer = torchreid.optim.build_optimizer(model, optim='sgd', lr=0.01)
        >>> # If you want to use a smaller learning rate for pretrained layers
        >>> # and the attribute name for the randomly initialized layer is 'classifier',
        >>> # you can do
        >>> optimizer = torchreid.optim.build_optimizer(
        >>>     model, optim='sgd', lr=0.01, staged_lr=True,
        >>>     new_layers='classifier', base_lr_mult=0.1
        >>> )
        >>> # Now the `classifier` has learning rate 0.01 but the base layers
        >>> # have learning rate 0.01 * 0.1.
        >>> # new_layers can also take multiple attribute names. Say the new layers
        >>> # are 'fc' and 'classifier', you can do
        >>> optimizer = torchreid.optim.build_optimizer(
        >>>     model, optim='sgd', lr=0.01, staged_lr=True,
        >>>     new_layers=['fc', 'classifier'], base_lr_mult=0.1
        >>> )
    优化器构建工具包装函数

    参数说明：
        model (nn.Module): 待训练的网络模型
        optim (str, 可选): 优化器类型，默认 "adam"
        lr (float, 可选): 基础学习率，默认 0.0003
        weight_decay (float, 可选): 权重衰减（L2正则惩罚系数），默认 5e-04
        momentum (float, 可选): SGD优化器的动量系数，默认 0.9
        sgd_dampening (float, 可选): SGD动量的阻尼系数，默认 0
        sgd_nesterov (bool, 可选): 是否开启Nesterov加速动量，默认 False
        rmsprop_alpha (float, 可选): RMSprop平滑系数，默认 0.99
        adam_beta1 (float, 可选): Adam一阶动量系数 β1，默认 0.9
        adam_beta2 (float, 可选): Adam二阶动量系数 β2，默认 0.99
        staged_lr (bool, 可选): 是否启用分层学习率策略。
            骨干层为预训练权重，新增层（如行人分类头）为随机初始化；
            开启后骨干层学习率 = lr * base_lr_mult，新增层使用完整lr。默认关闭。
        new_layers (str / list): 模型中新增层的属性名，默认为空
        base_lr_mult (float, 可选): 骨干网络学习率缩放倍数，默认 0.1

    使用示例::
        >>> # 普通优化器构建方式
        >>> optimizer = torchreid.optim.build_optimizer(model, optim='sgd', lr=0.01)

        >>> # 想给预训练骨干设置更小学习率，随机初始化层名为 classifier
        >>> optimizer = torchreid.optim.build_optimizer(
        >>>     model, optim='sgd', lr=0.01, staged_lr=True,
        >>>     new_layers='classifier', base_lr_mult=0.1
        >>> )
        >>> # 此时 classifier 层学习率为 0.01，骨干网络学习率为 0.01 * 0.1

        >>> # 多组新增层：fc、classifier 都使用完整学习率
        >>> optimizer = torchreid.optim.build_optimizer(
        >>>     model, optim='sgd', lr=0.01, staged_lr=True,
        >>>     new_layers=['fc', 'classifier'], base_lr_mult=0.1
        >>> )

    """
    if optim not in AVAI_OPTIMS:
        raise ValueError(
            'Unsupported optim: {}. Must be one of {}'.format(
                optim, AVAI_OPTIMS
            )
        )

    if not isinstance(model, nn.Module):
        raise TypeError(
            'model given to build_optimizer must be an instance of nn.Module'
        )
    #  开启分层学习率 staged_lr=True 分支
    if staged_lr:
        # 如果new_layers传的是单个字符串，转成列表统一处理
        if isinstance(new_layers, str):
            if new_layers is None:
                warnings.warn(
                    'new_layers is empty, therefore, staged_lr is useless'
                )
            new_layers = [new_layers]

        if isinstance(model, nn.DataParallel):
            model = model.module

        base_params = []
        base_layers = []
        new_params = []

        for name, module in model.named_children():
            if name in new_layers:
                new_params += [p for p in module.parameters()]
            else:
                base_params += [p for p in module.parameters()]
                base_layers.append(name)

        param_groups = [
            {
                'params': base_params,
                'lr': lr * base_lr_mult
            },
            {
                'params': new_params
            },
        ]

    else:
        param_groups = model.parameters()

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'amsgrad':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == 'radam':
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2)
        )
        
    elif optim == 'adamw':
        # AdamW 使用 torch.optim.AdamW，传参逻辑和Adam保持一致
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    return optimizer
