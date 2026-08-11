from __future__ import division, print_function, absolute_import
import csv
import os.path as osp
from collections import defaultdict

from ..dataset import ImageDataset


class ATRW(ImageDataset):
    """ATRW tiger re-identification dataset.

    Expected structure:
        reid-data/ATRW/atrw_reid_train/train/*.jpg
        reid-data/ATRW/atrw_anno_reid_train/reid_list_train.csv

    The CSV rows are:
        tiger_id,image_name

    The official training annotations are split deterministically by identity:
    60% identities for training and 40% identities for testing. For each test
    identity, one image is used as query and the rest are used as gallery.
    """

    dataset_dir = 'ATRW'
    dataset_url = None

    def __init__(self, root='', train_ratio=0.6, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)#这是一个方法，用于下载数据集，如果数据集不存在的话

        self.img_dir = osp.join(self.dataset_dir, 'atrw_reid_train', 'train')
        self.anno_dir = osp.join(self.dataset_dir, 'atrw_anno_reid_train')
        self.list_path = osp.join(self.anno_dir, 'reid_list_train.csv')
        self.train_ratio = train_ratio

        required_files = [self.dataset_dir, self.img_dir, self.list_path]
        self.check_before_run(required_files)#这是一个方法，用于检查数据集是否存在，如果不存在则抛出异常

        pid_to_imgs = self.read_annotations()
        train, query, gallery = self.build_split(pid_to_imgs)

        super(ATRW, self).__init__(train, query, gallery, **kwargs)

    def read_annotations(self):
        pid_to_imgs = defaultdict(list)
        with open(self.list_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                pid = int(row[0])
                img_name = row[1].strip()
                img_path = osp.join(self.img_dir, img_name)
                if not osp.isfile(img_path):
                    raise RuntimeError(
                        'ATRW image "{}" listed in "{}" is not found'.format(
                            img_path, self.list_path
                        )
                    )
                pid_to_imgs[pid].append(img_path)

        if not pid_to_imgs:
            raise RuntimeError('No ATRW annotations found in {}'.format(self.list_path))

        for pid in pid_to_imgs:
            pid_to_imgs[pid] = sorted(pid_to_imgs[pid])

        return pid_to_imgs

    def build_split(self, pid_to_imgs):
        pids = sorted(pid_to_imgs)
        num_train_pids = int(len(pids) * self.train_ratio)
        num_train_pids = max(1, min(num_train_pids, len(pids) - 1))

        train_pids = pids[:num_train_pids]
        test_pids = pids[num_train_pids:]
        pid2label = {pid: label for label, pid in enumerate(train_pids)}

        train = []
        for pid in train_pids:
            for img_path in pid_to_imgs[pid]:
                train.append((img_path, pid2label[pid], 0))

        query = []
        gallery = []
        for pid in test_pids:
            img_paths = pid_to_imgs[pid]
            query.append((img_paths[0], pid, 0))
            for img_path in img_paths[1:]:
                gallery.append((img_path, pid, 1))

        return train, query, gallery
