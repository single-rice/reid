import gc
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torchvision import models as reference_models

from torchreid.models import build_model
from torchreid.models._standard_utils import load_pretrained
from torchreid.models.vit import init_pretrained_weights as load_vit_weights
from torchreid.engine import ImageTripletEngine
from torchreid.utils import open_specified_layers, open_all_layers


class StandardModelsTest(unittest.TestCase):
    def test_head_only_training_freezes_top_level_parameters(self):
        model = torch.nn.Module()
        model.cls_token = torch.nn.Parameter(torch.ones(1, 1, 8))
        model.backbone = torch.nn.Linear(8, 8)
        model.head = torch.nn.Linear(8, 3)
        open_specified_layers(model, ['head'])
        self.assertFalse(model.cls_token.requires_grad)
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertTrue(model.head.weight.requires_grad)
        open_all_layers(model)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_torchvision_architectures_numerically_match(self):
        for name in ('efficientnet_b0', 'vit_b_16', 'swin_t', 'swin_s'):
            with self.subTest(model=name):
                reference = getattr(reference_models, name)(weights=None).eval()
                model = build_model(name, 1000, pretrained=False).eval()
                model.load_state_dict(reference.state_dict(), strict=True)
                image = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    torch.testing.assert_close(model.forward_classification(image), reference(image))
                del model, reference, image
                gc.collect()

    def test_native_dimensions_and_project_outputs(self):
        for name, dim in (('van_b2', 512), ('efficientnet_b0', 1280),
                          ('vit_b_16', 768), ('swin_t', 768), ('swin_s', 768),
                          ('conformer_base_patch16', 2112)):
            with self.subTest(model=name):
                model = build_model(name, 3, pretrained=False, loss='triplet')
                image = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    logits, features = model(image)
                    self.assertEqual(features.shape, (2, dim))
                    heads = logits if isinstance(logits, list) else [logits]
                    self.assertEqual(len(heads), 2 if name.startswith('conformer') else 1)
                    for head in heads:
                        self.assertEqual(head.shape, (2, 3))
                    model.loss = 'softmax'
                    softmax = model(image)
                    self.assertEqual(isinstance(softmax, list), name.startswith('conformer'))
                    model.eval()
                    self.assertEqual(model(image[:1]).shape, (1, dim))
                del model, image, logits, features, heads, head, softmax
                gc.collect()

    def test_transformers_reject_rectangular_input(self):
        for name in ('vit_b_16', 'swin_t', 'swin_s', 'conformer_base_patch16'):
            model = build_model(name, 3, pretrained=False).eval()
            with self.assertRaises((ValueError, AssertionError)):
                model(torch.randn(1, 3, 128, 256))
            del model
            gc.collect()

    def test_checkpoint_only_skips_mismatching_heads(self):
        # A complete ImageNet head may change size, but missing backbone keys fail.
        model = build_model('van_b0', 3, pretrained=False)
        state = dict(model.state_dict())
        state['head.weight'] = torch.randn(1000, model.feature_dim)
        state['head.bias'] = torch.randn(1000)
        with mock.patch('torch.hub.load_state_dict_from_url', return_value={'state_dict': state}):
            load_pretrained(model, 'https://example.invalid/van.pth', ('head.',))
        del state['patch_embed1.proj.weight']
        with mock.patch('torch.hub.load_state_dict_from_url', return_value={'state_dict': state}):
            with self.assertRaises(RuntimeError):
                load_pretrained(model, 'https://example.invalid/van.pth', ('head.',))

    def test_vit_legacy_checkpoint_names(self):
        model = build_model('vit_b_16', 3, pretrained=False)
        state = {k.replace('.mlp.0.', '.mlp.linear_1.').replace('.mlp.3.', '.mlp.linear_2.'): v
                 for k, v in model.state_dict().items()}
        with mock.patch('torch.hub.load_state_dict_from_url', return_value=state):
            load_vit_weights(model, 'https://example.invalid/vit.pth')

    def test_conformer_dual_head_training_engine(self):
        model = build_model('conformer_base_patch16', 2, pretrained=False, loss='triplet')
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        manager = SimpleNamespace(train_loader=[], test_loader={}, num_train_pids=2)
        engine = ImageTripletEngine(manager, model, optimizer, use_gpu=False,
                                    weight_t=.5, weight_x=1, weight_cv=1)
        result = engine.forward_backward({'img': torch.randn(4, 3, 224, 224),
                                         'pid': torch.tensor([0, 0, 1, 1]),
                                         'camid': torch.tensor([0, 1, 0, 0])})
        self.assertEqual(result['cross_video_anchor_fraction'], .5)
        for value in result.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))
        for layer in (model.conv_cls_head, model.trans_cls_head, model.trans_patch_conv, model.conv1):
            self.assertIsNotNone(layer.weight.grad)
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
