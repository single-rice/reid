from __future__ import absolute_import

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from torchreid import metrics
from torchreid.data.datasets.dataset import ImageDataset
from torchreid.data.datasets.image.wildtiger import WildTiger
from torchreid.engine.engine import evaluate_rank_by_query_group


class WildTigerEvaluationTest(unittest.TestCase):

    def test_query_csv_group_is_preserved_without_changing_tuple_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir)
            image_path = image_dir / 'tiger.jpg'
            image_path.touch()
            csv_path = image_dir / 'query.csv'
            with csv_path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['pid', 'image', 'camid', 'query'])
                writer.writerow([7, image_path.name, 2, 'multi'])

            rows, groups = WildTiger._read_rows(
                str(csv_path), str(image_dir), read_eval_group=True
            )

            self.assertEqual(len(rows[0]), 3)
            self.assertEqual(rows[0][1:], (7, 2))
            self.assertEqual(groups[rows[0][0]], 'multi')

    @mock.patch('torchreid.data.datasets.dataset.read_image')
    def test_image_dataset_exposes_optional_eval_group(self, read_image):
        read_image.return_value = object()
        sample = ('query.jpg', 0, 0)
        dataset = ImageDataset(
            [sample],
            [sample],
            [('gallery.jpg', 0, 1)],
            mode='query',
            verbose=False,
            eval_group_by_path={'query.jpg': 'sing'}
        )

        item = dataset[0]

        self.assertEqual(item['eval_group'], 'sing')
        self.assertEqual(len(dataset.query[0]), 4)

    def test_group_metrics_use_the_complete_shared_gallery(self):
        q_pids = np.asarray([0, 1, 2, 3])
        q_camids = np.asarray([0, 0, 1, 1])
        q_groups = np.asarray(['sing', 'sing', 'multi', 'multi'])
        g_pids = np.asarray([0, 1, 2, 2, 3, 3, 8, 9, 10, 11])
        g_camids = np.asarray([1, 1, 1, 2, 1, 2, 1, 1, 1, 1])
        distmat = np.asarray([
            [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.3, 0.4, 1.0],
            [0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.2, 0.3, 0.4, 1.0],
            [0.9, 0.8, 0.0, 0.2, 0.7, 0.6, 0.1, 0.3, 0.4, 1.0],
            [0.9, 0.8, 0.7, 0.6, 0.0, 0.2, 0.1, 0.3, 0.4, 1.0],
        ])

        overall_cmc, overall_map = metrics.evaluate_rank(
            distmat, q_pids, g_pids, q_camids, g_camids
        )
        results = evaluate_rank_by_query_group(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            q_groups
        )

        self.assertEqual(results['single_video']['num_queries'], 2)
        self.assertEqual(results['single_video']['num_entities'], 2)
        self.assertEqual(results['cross_video']['num_queries'], 2)
        self.assertEqual(results['cross_video']['num_entities'], 2)
        weighted_map = (
            results['single_video']['mAP'] * 2
            + results['cross_video']['mAP'] * 2
        ) / 4
        weighted_cmc = (
            results['single_video']['cmc'] * 2
            + results['cross_video']['cmc'] * 2
        ) / 4
        np.testing.assert_allclose(overall_map, weighted_map)
        # One same-camera positive is removed for each multi-video query, so
        # its maximum meaningful CMC rank is one shorter in this tiny fixture.
        np.testing.assert_allclose(overall_cmc[:9], weighted_cmc[:9])

    def test_group_count_must_match_query_count(self):
        with self.assertRaisesRegex(ValueError, 'one evaluation group per query'):
            evaluate_rank_by_query_group(
                np.zeros((2, 2)),
                np.asarray([0, 1]),
                np.asarray([0, 1]),
                np.asarray([0, 0]),
                np.asarray([1, 1]),
                np.asarray(['sing'])
            )


if __name__ == '__main__':
    unittest.main()
