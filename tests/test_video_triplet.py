import importlib.util
import random
import unittest
from collections import Counter
from pathlib import Path

import torch


def load(relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Sampler = load('torchreid/data/sampler.py').RandomIdentityVideoSampler
Loss = load('torchreid/losses/hard_mine_triplet_loss.py').CrossVideoTripletLoss


class VideoTripletTest(unittest.TestCase):
    def test_balanced_videos_and_identity_budget(self):
        random.seed(12)
        data = [('image', pid, cam) for pid in range(8)
                for cam in (range(2) if pid == 0 else range(1))
                for _ in range(8)]
        sampler = Sampler(data, 16, 8, 1.0)
        indices = list(sampler)
        self.assertEqual(len(indices) % 16, 0)
        counts = Counter(data[i][1] for i in indices)
        self.assertLessEqual(counts[0], 16)
        for start in range(0, len(indices), 16):
            batch = [data[i] for i in indices[start:start + 16]]
            self.assertEqual(sorted(Counter(x[1] for x in batch).values()), [8, 8])
            if any(x[1] == 0 for x in batch):
                self.assertEqual(Counter(x[2] for x in batch if x[1] == 0), {0: 4, 1: 4})

    def test_loss_excludes_same_video_positive(self):
        features = torch.tensor([[0.], [10.], [1.], [4.]], requires_grad=True)
        loss = Loss()(features, torch.tensor([0, 0, 0, 1]), torch.tensor([0, 0, 1, 0]))
        self.assertAlmostEqual(loss.item(), (0 + 3.3 + 6.3) / 3, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_no_valid_anchor_has_differentiable_zero(self):
        for pids in ([0, 0, 1], [0, 0, 0]):
            features = torch.randn(3, 2, requires_grad=True)
            loss = Loss()(features, torch.tensor(pids), torch.zeros(3, dtype=torch.long))
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_sampler_arguments(self):
        for batch, instances, probability in [(15, 8, .5), (8, 8, .5), (16, 8, 2)]:
            with self.assertRaises(ValueError):
                Sampler([], batch, instances, probability)
