from __future__ import absolute_import

import csv
import tempfile
import unittest
from pathlib import Path

from torchreid.data.datasets.image.atrw import ATRW


class ATRWEvaluationTest(unittest.TestCase):

    @staticmethod
    def write_query_csv(directory, query_group):
        image_path = directory / '000001.jpg'
        image_path.touch()
        csv_path = directory / 'query.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['pid', 'image', 'camid', 'query'])
            writer.writerow([7, image_path.name, 2, query_group])
        return csv_path, image_path

    def test_query_group_is_preserved_without_changing_tuple_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            csv_path, image_path = self.write_query_csv(directory, 'multi')

            rows, groups = ATRW._read_split(
                str(csv_path), str(directory), read_eval_group=True
            )

            self.assertEqual(rows, [(str(image_path), 7, 2)])
            self.assertEqual(len(rows[0]), 3)
            self.assertEqual(groups[str(image_path)], 'multi')

    def test_query_group_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            csv_path, image_path = self.write_query_csv(directory, ' SING ')

            _, groups = ATRW._read_split(
                str(csv_path), str(directory), read_eval_group=True
            )

            self.assertEqual(groups[str(image_path)], 'sing')

    def test_invalid_query_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            csv_path, _ = self.write_query_csv(directory, 'unknown')

            with self.assertRaisesRegex(RuntimeError, 'Invalid query group'):
                ATRW._read_split(
                    str(csv_path), str(directory), read_eval_group=True
                )


if __name__ == '__main__':
    unittest.main()
