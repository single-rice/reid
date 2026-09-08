"""Small initialization/checkpoint helpers; model blocks live in their own files."""

import inspect
import os
from pathlib import Path

import torch
from torch import nn


class DropPath(nn.Module):
    """Per-sample stochastic depth (the timm convention used by VAN/Conformer)."""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = (keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                                 dtype=x.dtype, device=x.device)).floor_()
        return x.div(keep) * mask


def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


def check_loss(loss):
    if loss not in ('softmax', 'triplet'):
        raise KeyError('Unsupported loss: {}'.format(loss))


def load_pretrained(model, source, classifier_prefixes, filename=None, key_mapping=None):
    """Require a complete backbone; skip only mismatching classification heads.

    Official state dictionaries, {'model': ...}, {'state_dict': ...} and a
    leading DataParallel 'module.' prefix are supported. Full classification
    heads load too when num_classes matches. Local files enable offline use.
    """
    source = os.fspath(source)
    load_options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_options['weights_only'] = True
    if 'drive.google.com' in source:
        import gdown
        path = Path(torch.hub.get_dir()) / 'checkpoints' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            downloaded = gdown.download(source, str(path), fuzzy=True, quiet=False)
            if downloaded is None:
                raise RuntimeError('Download failed; download manually and pass pretrained_path')
        checkpoint = torch.load(path, **load_options)
    elif source.startswith(('https://', 'http://')):
        checkpoint = torch.hub.load_state_dict_from_url(
            source, map_location='cpu', check_hash=True, file_name=filename)
    else:
        checkpoint = torch.load(source, **load_options)
    for key in ('state_dict', 'model', 'model_state_dict'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    state = {key[7:] if key.startswith('module.') else key: value
             for key, value in checkpoint.items()}
    if key_mapping is not None:
        state = {key_mapping(key): value for key, value in state.items()}
    current = model.state_dict()
    skipped = set()
    for key in list(state):
        if key.startswith(classifier_prefixes) and (
                key not in current or state[key].shape != current[key].shape):
            skipped.add(key)
            del state[key]
    missing = set(current) - set(state)
    unexpected = set(state) - set(current)
    mismatched = {key for key in set(state) & set(current)
                  if state[key].shape != current[key].shape}
    if missing - skipped or unexpected or mismatched:
        raise RuntimeError('Incompatible pretrained backbone: missing={}, unexpected={}, shapes={}'.format(
            sorted(missing - skipped), sorted(unexpected), sorted(mismatched)))
    model.load_state_dict(state, strict=False)
    print('Loaded pretrained weights: {} (reinitialized heads: {})'.format(source, sorted(skipped)))
