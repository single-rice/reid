from __future__ import absolute_import, division, print_function

import csv
import os.path as osp

from ..dataset import ImageDataset


class ATRW(ImageDataset):
    """ATRW tiger re-identification dataset.

    Run ``python tools/prepare_atrw.py`` before using this loader. The prepared
    dataset has the following structure::

        ATRW/
          train/       # all official training images
          query/       # two images per official test identity
          gallery/     # the remaining official test images
          train.csv
          query.csv
          gallery.csv

    Test CSV files contain ``pid,image,camid,query``. For ``sing`` identities,
    query images use camid 0 and gallery images use camid 1. For ``multi``
    identities, images from the same source video share a camid in both splits;
    the first source uses camid 1 and the remaining source(s) use camid 2.
    """

    dataset_dir = 'ATRW'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.train_csv = osp.join(self.dataset_dir, 'train.csv')
        self.query_csv = osp.join(self.dataset_dir, 'query.csv')
        self.gallery_csv = osp.join(self.dataset_dir, 'gallery.csv')

        self.check_before_run([
            self.train_dir, self.query_dir, self.gallery_dir,
            self.train_csv, self.query_csv, self.gallery_csv
        ])

        train = self._read_split(self.train_csv, self.train_dir, relabel=True,
                                 default_camid=0)
        query, eval_group_by_path = self._read_split(
            self.query_csv, self.query_dir, read_eval_group=True
        )
        gallery = self._read_split(self.gallery_csv, self.gallery_dir)
        super(ATRW, self).__init__(
            train,
            query,
            gallery,
            eval_group_by_path=eval_group_by_path,
            **kwargs
        )

    @staticmethod
    def _read_split(
        csv_path,
        image_dir,
        relabel=False,
        default_camid=None,
        read_eval_group=False
    ):
        rows = []
        eval_group_by_path = {}
        with open(csv_path, 'r', encoding='utf-8', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or not {
                'pid', 'image'
            }.issubset(reader.fieldnames):
                raise RuntimeError(
                    'Invalid ATRW split file: {}'.format(csv_path)
                )
            for row in reader:
                pid = int(row['pid'])
                if 'camid' in row and row['camid'].strip():
                    camid = int(row['camid'])
                elif default_camid is not None:
                    camid = default_camid
                else:
                    raise RuntimeError(
                        'Missing camid in ATRW split: {}'.format(csv_path)
                    )
                image_path = osp.join(image_dir, row['image'].strip())
                if not osp.isfile(image_path):
                    raise RuntimeError(
                        'ATRW image not found: {}'.format(image_path)
                    )
                rows.append((image_path, pid, camid))
                if read_eval_group:
                    if 'query' not in row or not row['query'].strip():
                        raise RuntimeError(
                            'Missing query group in ATRW split: {}'.format(
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
            raise RuntimeError('ATRW split is empty: {}'.format(csv_path))

        pid2label = {}
        if relabel:
            pid2label = {
                pid: label
                for label, pid in enumerate(sorted({x[1] for x in rows}))
            }
        data = [
            (path, pid2label.get(pid, pid), camid)
            for path, pid, camid in rows
        ]
        if read_eval_group:
            return data, eval_group_by_path
        return data
