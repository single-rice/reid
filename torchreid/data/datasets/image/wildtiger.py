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
    403 个身份、17,669 张图片
    数据划分结果：
    训练集：242 个 ID，10,466 张图片
    验证集：81 个 ID，3,606 张图片
    测试集：80 个 ID
    每个测试 ID 的每个视频随机选择 2 张图片放入 Query。
    该视频其余图片全部放入 Gallery。
    Query：184 张
    Gallery：3,413 张
    单视频 ID：71 个
    多视频 ID：9 个

    Query 全部使用 camid=0。
    Gallery 按每个 ID 内排序后的视频编号：
    第一个视频：camid=1
    第二个视频：camid=2

    随机种子：42
    | 老虎 ID | 视频数 | 视频 |
    |---|---:|---|
    | tiger_028 | 2 | video_028、video_029 |
    | tiger_125 | 2 | video_124、video_125 |
    | tiger_145 | 2 | video_145、video_146 |
    | tiger_181 | 2 | video_181、video_192 |
    | tiger_274 | 2 | video_274、video_292 |
    | tiger_276 | 2 | video_276、video_279 |
    | tiger_392 | 4 | video_392、video_396、video_412、video_460 |
    | tiger_408 | 3 | video_408、video_409、video_410 |
    | tiger_479 | 2 | video_479、video_480 |
    
    遵循 torchreid 数据集基类规范，输出 (图片路径, pid行人ID, camid摄像头ID) 三元组给 ImageDataset
    datamanager = ImageDataManager(
    root=r'D:\code\deep-person-reid-master\reid-data',
    sources='wildtiger',
    targets='wildtiger'
    )

    train_loader = datamanager.train_loader
    val_loader = datamanager.val_loader['wildtiger']
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
            video2camid = {
                video: index + 1 for index, video in enumerate(video_names)
            }
            for image_path in sorted(image_paths):
                camid = (
                    video2camid[self.get_video_name(image_path)]
                    if split == 'gallery' else 0
                )
                data.append((image_path, pid, camid))
        if not data:
            raise RuntimeError('No images found in {}'.format(directory))
        return data

    @staticmethod
    def get_video_name(image_path):
        match = re.search(
            r'(video_\d+)', osp.basename(image_path), re.IGNORECASE
        )
        if match is None:
            raise RuntimeError(
                'Cannot parse video name from "{}"; expected '
                'video_<number>'.format(osp.basename(image_path))
            )
        return match.group(1).lower()
