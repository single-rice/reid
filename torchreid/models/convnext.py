"""ConvNeXt backbones with a lightweight global Transformer head.

The convolutional trunk follows the ConvNeXt-T/S design (large-kernel
depthwise convolution, channel-last LayerNorm and inverted bottleneck MLP).
At the final, low-resolution feature map a small spatial Transformer is added
so that the model keeps ConvNeXt's local texture bias while also modelling
long-range relations between body parts.

The ImageNet checkpoints published by torchvision are compatible with the
convolutional part of this implementation.  The Transformer head and the
re-identification classifier are intentionally left randomly initialized.
"""

from __future__ import absolute_import

import os
import re
import warnings

import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F


__all__ = ['convnext_tiny', 'convnext_small', 'convnext_t', 'convnext_s']


model_urls = {
    # torchvision ConvNeXt ImageNet-1K checkpoints
    'convnext_tiny':
    'https://download.pytorch.org/models/convnext_tiny-983f1562.pth',
    'convnext_small':
    'https://download.pytorch.org/models/convnext_small-0c510722.pth',
}


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for a BCHW tensor."""

    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Per-sample stochastic depth, kept local to avoid extra dependencies."""

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    """A ConvNeXt block with names compatible with the official checkpoint."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # torchvision stores layer_scale as [C, 1, 1], which also lets the
        # official ConvNeXt checkpoint load this parameter directly.
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim, 1, 1)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma.view(1, 1, 1, -1) * x
        x = x.permute(0, 3, 1, 2)
        return identity + self.drop_path(x)


class GlobalTransformerBlock(nn.Module):
    """Pre-norm spatial self-attention block operating on BCHW features."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super(GlobalTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Sequence-first layout works with old and new PyTorch releases.
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        tokens = x.flatten(2).permute(2, 0, 1)
        normalized = self.norm1(tokens)
        attended, _ = self.attn(normalized, normalized, normalized)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.permute(1, 2, 0).reshape(
            batch, channels, height, width
        )


class ConvNeXtHybrid(nn.Module):
    """ConvNeXt-T/S plus low-resolution global self-attention.

    Args:
        depths: Number of ConvNeXt blocks in the four stages.
        dims: Channel widths of the four stages.
        transformer_depth: Number of global attention blocks at stage 4.
        last_stride: Stride of the final downsampling layer.  ``1`` keeps
            more spatial detail, which is usually useful for person Re-ID.
    """

    def __init__(
        self,
        num_classes,
        loss,
        depths,
        dims,
        transformer_depth=2,
        transformer_heads=8,
        last_stride=1,
        drop_path_rate=0.1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    ):
        super(ConvNeXtHybrid, self).__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError('depths and dims must contain four stages')
        if last_stride not in (1, 2):
            raise ValueError('last_stride must be 1 or 2')

        self.loss = loss
        self.feature_dim = dims[-1]
        self.dims = dims

        # This ordering mirrors torchvision ConvNeXt's features sequence:
        # stem, stage 1, downsample, stage 2, downsample, ...
        self.features = nn.ModuleList()
        self.features.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0])
            )
        )
        total_blocks = sum(depths)
        block_index = 0
        for stage_index, (depth, dim) in enumerate(zip(depths, dims)):
            drop_rates = [
                drop_path_rate * block_index / max(total_blocks - 1, 1)
                for block_index in range(
                    block_index, block_index + depth
                )
            ]
            self.features.append(
                nn.Sequential(*[
                    ConvNeXtBlock(dim, drop_path=drop_rate)
                    for drop_rate in drop_rates
                ])
            )
            block_index += depth
            if stage_index < 3:
                # The last_stride switch controls the downsampling between
                # stage 3 and stage 4, matching the Re-ID convention.
                stride = last_stride if stage_index == 2 else 2
                self.features.append(
                    nn.Sequential(
                        LayerNorm2d(dim),
                        nn.Conv2d(
                            dim, dims[stage_index + 1], 2, stride=stride
                        )
                    )
                )

        # At 256x128, the default last_stride=1 gives only 16x8 tokens.
        # Attention therefore adds global context at a modest computational
        # cost compared with applying it to the input or an early stage.
        self.pos_embed = nn.Parameter(torch.zeros(1, dims[-1], 8, 4))
        self.global_transformer = nn.Sequential(*[
            GlobalTransformerBlock(dims[-1], transformer_heads)
            for _ in range(transformer_depth)
        ])
        self.head_norm = nn.LayerNorm(dims[-1])
        self.fc = self._construct_fc_layer(fc_dims, dims[-1], dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        if not isinstance(fc_dims, (list, tuple)):
            raise TypeError('fc_dims must be either list or tuple')
        layers = []
        for dim in fc_dims:
            layers.extend([nn.Linear(input_dim, dim), nn.BatchNorm1d(dim)])
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, LayerNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.pos_embed, std=0.02)

    def featuremaps(self, x):
        # features = stem, stage/downsample pairs, final stage.
        for module in self.features:
            x = module(x)
        x = x + F.interpolate(
            self.pos_embed,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        return self.global_transformer(x)

    def forward(self, x):
        f = self.featuremaps(x)
        v = f.mean(dim=(2, 3))
        v = self.head_norm(v)
        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        if self.loss == 'triplet':
            return y, v
        raise KeyError('Unsupported loss: {}'.format(self.loss))


def _clean_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError('Pretrained checkpoint must contain a state dict')
    for key in ('state_dict', 'model', 'model_state_dict'):
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def _remap_checkpoint_key(key):
    """Map common torchvision ConvNeXt names to this model's names."""
    key = re.sub(r'^(model\.|backbone\.)', '', key)
    key = key.replace('.layer_scale', '.gamma')
    key = key.replace('.block.0.', '.dwconv.')
    key = key.replace('.block.2.', '.norm.')
    key = key.replace('.block.3.', '.pwconv1.')
    key = key.replace('.block.5.', '.pwconv2.')
    key = key.replace('classifier.0.', 'head_norm.')
    return key


def _checkpoint_key_candidates(key):
    """Return compatible names for both ConvNeXt norm/conv orderings."""
    remapped = _remap_checkpoint_key(key)
    candidates = [remapped]
    # torchvision versions have used both Conv-Norm and Norm-Conv in the
    # downsampling modules. The shape check selects the right one.
    match = re.match(
        r'^(features\.(?:2|4|6))\.(0|1)\.(weight|bias)$', remapped
    )
    if match:
        prefix, index, parameter = match.groups()
        other_index = '1' if index == '0' else '0'
        candidates.append(
            '{}.{}.{}'.format(prefix, other_index, parameter)
        )
    return candidates


def init_pretrained_weights(model, model_url):
    """Load compatible ImageNet weights, skipping the Re-ID/Transformer head."""
    if os.path.isfile(model_url):
        checkpoint = torch.load(model_url, map_location='cpu')
    else:
        checkpoint = model_zoo.load_url(model_url, map_location='cpu')
    checkpoint = _clean_state_dict(checkpoint)
    model_dict = model.state_dict()
    compatible = {}
    for checkpoint_key, value in checkpoint.items():
        for key in _checkpoint_key_candidates(checkpoint_key):
            if key in model_dict and model_dict[key].size() == value.size():
                compatible[key] = value
                break
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    if not compatible:
        warnings.warn(
            'No compatible ConvNeXt weights found in {}'.format(model_url)
        )
    return len(compatible)


def _build_convnext(
    name,
    num_classes,
    loss='softmax',
    pretrained=True,
    **kwargs
):
    model = ConvNeXtHybrid(
        num_classes=num_classes,
        loss=loss,
        depths=[3, 3, 9, 3] if name == 'convnext_tiny' else [3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        **kwargs
    )
    if pretrained:
        url_or_path = pretrained if isinstance(pretrained, str) else model_urls[name]
        init_pretrained_weights(model, url_or_path)
    return model


def convnext_tiny(num_classes, loss='softmax', pretrained=True, **kwargs):
    """ConvNeXt-T with a two-block global Transformer head."""
    return _build_convnext(
        'convnext_tiny', num_classes, loss, pretrained, **kwargs
    )


def convnext_small(num_classes, loss='softmax', pretrained=True, **kwargs):
    """ConvNeXt-S with a two-block global Transformer head."""
    return _build_convnext(
        'convnext_small', num_classes, loss, pretrained, **kwargs
    )


# Short aliases are useful in experiment configuration files that refer to
# the variants as ConvNeXt-T/S rather than Tiny/Small.
convnext_t = convnext_tiny
convnext_s = convnext_small
