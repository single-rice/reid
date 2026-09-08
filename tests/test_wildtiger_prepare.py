from __future__ import absolute_import

import random
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.prepare_wildtiger import (
    build_test_records,
    split_identities,
    split_query_gallery,
    video_names,
    main,
)


class WildTigerPreparationTest(unittest.TestCase):

    def test_generated_training_video_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            for pid in range(3):
                folder = source / ('tiger_%03d' % pid)
                folder.mkdir(parents=True)
                for video in range(2):
                    for frame in range(5):
                        (folder / ('video_%03d_%03d.jpg' % (video, frame))).touch()
            destination = root / 'prepared'
            argv = ['prepare', '--source', str(source), '--destination', str(destination),
                    '--test-multi-ids', '0', '--test-single-ids', '0']
            with mock.patch('sys.argv', argv):
                main()
            for split in ('train', 'val'):
                with (destination / (split + '.csv')).open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertTrue(rows)
                for row in rows:
                    self.assertIn(row['video_id'], row['image'])
                    self.assertEqual(int(row['camid']), int(row['video_id'].split('_')[1]) + 1)

    @staticmethod
    def identity(name, videos, images_per_video=3):
        images = []
        for video in videos:
            images.extend(
                Path('video_{:03d}_{:02d}.jpg'.format(video, index))
                for index in range(images_per_video)
            )
        return name, images

    def test_stratified_identity_split(self):
        identities = [
            self.identity('multi_{:02d}'.format(index), [index, index + 100])
            for index in range(4)
        ] + [
            self.identity('single_{:02d}'.format(index), [index + 200])
            for index in range(6)
        ]

        train_val, test = split_identities(
            identities, random.Random(42), num_multi=3, num_single=5
        )

        self.assertEqual(len(train_val), 2)
        self.assertEqual(len(test), 8)
        self.assertEqual(
            sum(len(video_names(images)) > 1 for _, images in test), 3
        )
        self.assertEqual(
            sum(len(video_names(images)) == 1 for _, images in test), 5
        )
        self.assertEqual(
            {name for name, _ in train_val}.intersection(
                name for name, _ in test
            ),
            set()
        )

    def test_query_split_selects_two_images_per_identity(self):
        _, images = self.identity('multi', [1, 2], images_per_video=4)
        query, gallery, counts = split_query_gallery(
            images, random.Random(42)
        )

        self.assertEqual(len(query), 2)
        self.assertEqual(len(gallery), 6)
        self.assertEqual(counts, {'video_001': 4, 'video_002': 4})
        self.assertFalse(set(query).intersection(gallery))

    def test_single_video_uses_atrw_pseudo_camids(self):
        _, images = self.identity('single', [7], images_per_video=4)
        query, gallery = images[:2], images[2:]
        query_rows, gallery_rows, mapping = build_test_records(
            7, query, gallery
        )

        self.assertEqual(mapping, {'video_007': 1})
        self.assertEqual({row[2] for row in query_rows}, {0})
        self.assertEqual({row[2] for row in gallery_rows}, {1})
        self.assertEqual({row[3] for row in query_rows + gallery_rows}, {'sing'})

    def test_multi_video_keeps_all_distinct_camids(self):
        _, images = self.identity('multi', [1, 2, 3, 4], images_per_video=3)
        query = [images[0], images[4]]
        gallery = [image for image in images if image not in query]
        query_rows, gallery_rows, mapping = build_test_records(
            9, query, gallery
        )

        self.assertEqual(
            mapping,
            {
                'video_001': 1,
                'video_002': 2,
                'video_003': 3,
                'video_004': 4
            }
        )
        all_rows = query_rows + gallery_rows
        self.assertEqual({row[2] for row in all_rows}, {1, 2, 3, 4})
        self.assertEqual({row[3] for row in all_rows}, {'multi'})

    def test_rejects_unavailable_identity_quota(self):
        identities = [self.identity('multi', [1, 2])]
        with self.assertRaisesRegex(RuntimeError, 'only 1 are available'):
            split_identities(
                identities, random.Random(42), num_multi=2, num_single=0
            )


if __name__ == '__main__':
    unittest.main()
