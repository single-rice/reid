from __future__ import division, print_function, absolute_import
import glob
import os.path as osp
import re

from ..dataset import ImageDataset


class TigerMini(ImageDataset):
    """TigerMini.

    Directory structure:
        reid-data/TigerMini/train
        reid-data/TigerMini/query
        reid-data/TigerMini/gallery

    File name format:
        video_XXX_YYYY.jpg

    XXX is treated as the tiger identity. Since the dataset currently has no
    real camera labels, the minimal evaluation-friendly camid rule is used:
    train/query -> 0, gallery -> 1.
    """

    dataset_dir = 'TigerMini'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True, camid=0)
        query = self.process_dir(self.query_dir, relabel=False, camid=0)
        gallery = self.process_dir(self.gallery_dir, relabel=False, camid=1)

        super(TigerMini, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False, camid=0):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'^video_(\d+)_(\d+)$')

        pid_container = set()
        parsed = []
        for img_path in img_paths:
            fname = osp.splitext(osp.basename(img_path))[0]
            match = pattern.match(fname)
            if match is None:
                raise RuntimeError(
                    'Invalid TigerMini image name "{}"; expected '
                    '"video_XXX_YYYY.jpg"'.format(osp.basename(img_path))
                )
            pid = int(match.group(1))
            pid_container.add(pid)
            parsed.append((img_path, pid))

        pid2label = {
            pid: label for label, pid in enumerate(sorted(pid_container))
        }

        data = []
        for img_path, pid in parsed:
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
