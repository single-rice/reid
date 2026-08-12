from __future__ import absolute_import, division, print_function

import glob
import os.path as osp
import re

from ..dataset import ImageDataset


class WildTiger(ImageDataset):
    """Wild_Tiger image re-identification dataset.

    Expected layout::

        Wild_Tiger/train/<tiger_id>/*
        Wild_Tiger/val/<tiger_id>/*
        Wild_Tiger/test/query/<tiger_id>/*
        Wild_Tiger/test/gallery/<tiger_id>/*

    Validation data is retained on disk for model selection. Torchreid's
    standard interface consumes train, query and gallery; query/gallery are
    therefore read from the test directory.
    """

    dataset_dir = 'Wild_Tiger'
    dataset_url = None
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.query_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')
        self.check_before_run([
            self.dataset_dir, self.train_dir, self.val_dir,
            self.query_dir, self.gallery_dir
        ])

        train = self.process_dir(self.train_dir, relabel=True, split='train')
        self.val = self.process_dir(self.val_dir, relabel=False, split='val')
        query = self.process_dir(self.query_dir, relabel=False, split='query')
        gallery = self.process_dir(self.gallery_dir, relabel=False, split='gallery')
        super(WildTiger, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, directory, relabel=False, split='train'):
        tiger_dirs = sorted(path for path in glob.glob(osp.join(directory, '*')) if osp.isdir(path))
        pid_names = [osp.basename(path) for path in tiger_dirs]
        pid2label = {pid: label for label, pid in enumerate(pid_names)}
        data = []
        for tiger_dir in tiger_dirs:
            pid_name = osp.basename(tiger_dir)
            match = re.match(r'^tiger_(\d+)$', pid_name, re.IGNORECASE)
            if match is None:
                raise RuntimeError(
                    'Invalid identity directory "{}"; expected tiger_<number>'.format(pid_name)
                )
            pid = pid2label[pid_name] if relabel else int(match.group(1))
            image_paths = []
            for extension in self.image_extensions:
                image_paths.extend(glob.glob(osp.join(tiger_dir, extension)))
            video_names = sorted({self.get_video_name(path) for path in image_paths})
            video2camid = {video: index + 1 for index, video in enumerate(video_names)}
            for image_path in sorted(image_paths):
                camid = video2camid[self.get_video_name(image_path)] if split == 'gallery' else 0
                data.append((image_path, pid, camid))
        if not data:
            raise RuntimeError('No images found in {}'.format(directory))
        return data

    @staticmethod
    def get_video_name(image_path):
        match = re.search(r'(video_\d+)', osp.basename(image_path), re.IGNORECASE)
        if match is None:
            raise RuntimeError(
                'Cannot parse video name from "{}"; expected video_<number>'.format(
                    osp.basename(image_path)
                )
            )
        return match.group(1).lower()
