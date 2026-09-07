from __future__ import absolute_import, division, print_function

import csv
import os.path as osp

from ..dataset import ImageDataset


class WildTiger(ImageDataset):
    """WildTiger image re-identification dataset.

    Run ``tools/prepare_wildtiger.py`` before using this loader. The prepared
    dataset contains train/validation identity folders, test query/gallery
    folders and four CSV split files. CSV image paths are relative to their
    corresponding split directory.

    Test CSV files contain ``pid,image,camid,query``. For ``sing`` identities,
    query images use the evaluation camid 0 and gallery images use camid 1.
    For ``multi`` identities, videos are sorted per identity and assigned
    distinct camids starting at 1; images from the same video retain the same
    camid in query and gallery.
    | 划分 | ID 数 | 图片数 |
    | train | 323 | 11026 |
    | val | 323 | 2763 |
    | test/query | 80 | 160 |
    | test/gallery | 80 | 3720 |
    多视频身份：30
    单视频身份：50
    合计：80
    """

    dataset_dir = 'Wild_Tiger'
    dataset_url = None
    has_validation = True

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.query_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')
        self.train_csv = osp.join(self.dataset_dir, 'train.csv')
        self.val_csv = osp.join(self.dataset_dir, 'val.csv')
        self.query_csv = osp.join(self.dataset_dir, 'query.csv')
        self.gallery_csv = osp.join(self.dataset_dir, 'gallery.csv')
        self.check_before_run([
            self.dataset_dir,
            self.train_dir,
            self.val_dir,
            self.query_dir,
            self.gallery_dir,
            self.train_csv,
            self.val_csv,
            self.query_csv,
            self.gallery_csv
        ])

        train_rows = self._read_rows(
            self.train_csv, self.train_dir, default_camid=0
        )
        pid2label = {
            pid: label
            for label, pid in enumerate(sorted({item[1] for item in train_rows}))
        }
        train = self._relabel(train_rows, pid2label, 'train')
        val = self._relabel(
            self._read_rows(self.val_csv, self.val_dir, default_camid=0),
            pid2label,
            'val'
        )
        query, eval_group_by_path = self._read_rows(
            self.query_csv, self.query_dir, read_eval_group=True
        )
        gallery = self._read_rows(self.gallery_csv, self.gallery_dir)
        super(WildTiger, self).__init__(
            train,
            query,
            gallery,
            val=val,
            eval_group_by_path=eval_group_by_path,
            **kwargs
        )

    @staticmethod
    def _read_rows(
        csv_path, image_dir, default_camid=None, read_eval_group=False
    ):
        rows = []
        eval_group_by_path = {}
        with open(csv_path, 'r', encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not {
                'pid', 'image'
            }.issubset(reader.fieldnames):
                raise RuntimeError(
                    'Invalid WildTiger split file: {}'.format(csv_path)
                )
            for row in reader:
                pid = int(row['pid'])
                if 'camid' in row and row['camid'].strip():
                    camid = int(row['camid'])
                elif default_camid is not None:
                    camid = default_camid
                else:
                    raise RuntimeError(
                        'Missing camid in WildTiger split: {}'.format(csv_path)
                    )
                image_path = osp.join(image_dir, *row['image'].split('/'))
                if not osp.isfile(image_path):
                    raise RuntimeError(
                        'WildTiger image not found: {}'.format(image_path)
                    )
                rows.append((image_path, pid, camid))
                if read_eval_group:
                    if 'query' not in row or not row['query'].strip():
                        raise RuntimeError(
                            'Missing query group in WildTiger split: {}'.format(
                                csv_path
                            )
                        )
                    eval_group = row['query'].strip().lower()
                    if eval_group not in {'sing', 'multi'}:
                        raise RuntimeError(
                            'Invalid query group "{}" in {}'.format(
                                eval_group, csv_path
                            )
                        )
                    eval_group_by_path[image_path] = eval_group

        if not rows:
            raise RuntimeError('WildTiger split is empty: {}'.format(csv_path))
        if read_eval_group:
            return rows, eval_group_by_path
        return rows

    @staticmethod
    def _relabel(rows, pid2label, split):
        data = []
        for path, pid, camid in rows:
            if pid not in pid2label:
                raise RuntimeError(
                    'Identity {} in {} is absent from the training split'.format(
                        pid, split
                    )
                )
            data.append((path, pid2label[pid], camid))
        return data
