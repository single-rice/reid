from __future__ import division, absolute_import
import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler

AVAI_SAMPLERS = [
    'RandomIdentitySampler', 'SequentialSampler', 'RandomSampler',
    'RandomDomainSampler', 'RandomDatasetSampler', 'RandomIdentityVideoSampler'
]


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    随机采样 N 个行人身份，每个身份采样 K 张图片。

    参数说明：
        data_source (list): 数据集列表，每个元素为元组 (图片路径, 行人ID, 摄像头ID, 数据集编号)
        batch_size (int): 批次总样本数量
        num_instances (int): 单个行人身份在一个批次内采样的图片数量
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, items in enumerate(data_source):
            pid = items[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentityVideoSampler(RandomIdentitySampler):
    """Use camid as video ID within each identity.

    Identity chunk quotas match RandomIdentitySampler, without replenishment.
    With the configured probability, reserve one identity slot for an available
    multi-video identity. Other slots are sampled normally. Multi-video chunks
    always draw equally (within one image) from two randomly selected videos.
    The reservation is best effort once multi-video quotas are exhausted.
    """

    def __init__(self, data_source, batch_size, num_instances,
                 video_batch_probability=0.5):
        if num_instances < 2 or batch_size % num_instances:
            raise ValueError('Require num_instances >= 2 and divisible batch_size')
        if batch_size // num_instances < 2:
            raise ValueError('Triplet batches require at least two identities')
        if not 0 <= video_batch_probability <= 1:
            raise ValueError('video_batch_probability must be in [0, 1]')
        super().__init__(data_source, batch_size, num_instances)
        self.video_batch_probability = video_batch_probability
        self.videos = defaultdict(lambda: defaultdict(list))
        for index, item in enumerate(data_source):
            self.videos[item[1]][item[2]].append(index)

    def __iter__(self):
        chunks = {}
        for pid, indices in self.index_dic.items():
            indices = list(indices)
            random.shuffle(indices)
            count = max(1, len(indices) // self.num_instances)
            chunks[pid] = []
            for chunk in range(count):
                videos = self.videos[pid]
                if len(videos) >= 2:
                    chosen = random.sample(list(videos), 2)
                    group = []
                    for i, video in enumerate(chosen):
                        size = self.num_instances // 2 + (i < self.num_instances % 2)
                        pool = videos[video]
                        group.extend(random.sample(pool, min(size, len(pool))))
                        if size > len(pool):
                            group.extend(random.choices(pool, k=size - len(pool)))
                else:
                    group = indices[chunk * self.num_instances:(chunk + 1) * self.num_instances]
                    group += random.choices(indices, k=self.num_instances - len(group))
                chunks[pid].append(group)
        available = list(self.pids)
        result = []
        while len(available) >= self.num_pids_per_batch:
            multi = [pid for pid in available if len(self.videos[pid]) >= 2]
            chosen = []
            if multi and random.random() < self.video_batch_probability:
                chosen.append(random.choice(multi))
            chosen += random.sample([pid for pid in available if pid not in chosen],
                                    self.num_pids_per_batch - len(chosen))
            for pid in chosen:
                result.extend(chunks[pid].pop())
                if not chunks[pid]:
                    available.remove(pid)
        return iter(result)


class RandomDomainSampler(Sampler):
    """Random domain sampler.

    We consider each camera as a visual domain.

    How does the sampling work:
    1. Randomly sample N cameras (based on the "camid" label).
    2. From each camera, randomly sample K images.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        n_domain (int): number of cameras to sample in a batch.
        
    随机域采样器

    本采样器将每一个摄像头视作一个独立视觉域。

    采样流程：
    1. 基于摄像头编号 camid，随机抽取 N 个不同摄像头；
    2. 从选中的每个摄像头中，随机抽取 K 张图片。

    参数说明：
        data_source (list): 数据集列表，每条数据元组格式 (图片路径, 行人ID, 摄像头ID, 数据集编号)
        batch_size (int): 批次总图片数量
        n_domain (int): 单个批次内采样的摄像头（域）数量

    """

    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, items in enumerate(data_source):
            camid = items[2]
            self.domain_dict[camid].append(i)
        self.domains = list(self.domain_dict.keys())

        # Make sure each domain can be assigned an equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomDatasetSampler(Sampler):
    """Random dataset sampler.

    How does the sampling work:
    1. Randomly sample N datasets (based on the "dsetid" label).
    2. From each dataset, randomly sample K images.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        n_dataset (int): number of datasets to sample in a batch.
        多数据集随机采样器

    采样流程：
    1. 根据数据集编号 dsetid，随机选取 N 个不同数据集；
    2. 从选中的每个数据集里随机抽取 K 张图片。

    参数说明：
        data_source (list): 数据列表，每条是元组 (图片路径, 行人ID, 摄像头ID, 数据集编号)
        batch_size (int): 批次总样本数
        n_dataset (int): 一个批次内要采样的数据集个数
    """

    def __init__(self, data_source, batch_size, n_dataset):
        self.data_source = data_source

        # Keep track of image indices for each dataset
        self.dataset_dict = defaultdict(list)
        for i, items in enumerate(data_source):
            dsetid = items[3]
            self.dataset_dict[dsetid].append(i)
        self.datasets = list(self.dataset_dict.keys())

        # Make sure each dataset can be assigned an equal number of images
        if n_dataset is None or n_dataset <= 0:
            n_dataset = len(self.datasets)
        assert batch_size % n_dataset == 0
        self.n_img_per_dset = batch_size // n_dataset

        self.batch_size = batch_size
        self.n_dataset = n_dataset
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        dataset_dict = copy.deepcopy(self.dataset_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_datasets = random.sample(self.datasets, self.n_dataset)

            for dset in selected_datasets:
                idxs = dataset_dict[dset]
                selected_idxs = random.sample(idxs, self.n_img_per_dset)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    dataset_dict[dset].remove(idx)

                remaining = len(dataset_dict[dset])
                if remaining < self.n_img_per_dset:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_sampler(
    data_source,
    train_sampler,
    batch_size=32,
    num_instances=4,
    num_cams=1,
    num_datasets=1,
    **kwargs
):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (when using ``RandomIdentitySampler``). Default is 4.
        num_cams (int, optional): number of cameras to sample in a batch (when using
            ``RandomDomainSampler``). Default is 1.
        num_datasets (int, optional): number of datasets to sample in a batch (when
            using ``RandomDatasetSampler``). Default is 1.
    构建训练集采样器

    参数说明：
        data_source (list): 训练数据列表，每条为元组 (图片路径, 行人ID, 摄像头ID)
        train_sampler (str): 采样器名称，默认 RandomSampler（普通随机采样）
        batch_size (int, 可选): 批次大小，默认32
        num_instances (int, 可选):
            每个行人ID在单批次内采样图片数量，仅 RandomIdentitySampler 生效，默认4
        num_cams (int, 可选):
            单批次采样摄像头数量，仅 RandomDomainSampler 生效，默认1
        num_datasets (int, 可选):
            单批次采样数据集数量，仅 RandomDatasetSampler 生效，默认1
    """
    assert train_sampler in AVAI_SAMPLERS, \
        'train_sampler must be one of {}, but got {}'.format(AVAI_SAMPLERS, train_sampler)

    if train_sampler == 'RandomIdentitySampler':#按行人身份采样（ReID三元组训练标配）
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)

    elif train_sampler == 'RandomIdentityVideoSampler':
        sampler = RandomIdentityVideoSampler(
            data_source, batch_size, num_instances,
            kwargs.get('video_batch_probability', 0.5))

    elif train_sampler == 'RandomDomainSampler':
        sampler = RandomDomainSampler(data_source, batch_size, num_cams)

    elif train_sampler == 'RandomDatasetSampler':
        sampler = RandomDatasetSampler(data_source, batch_size, num_datasets)

    elif train_sampler == 'SequentialSampler':#顺序采样（不打乱，按文件顺序读取）
        sampler = SequentialSampler(data_source)

    elif train_sampler == 'RandomSampler':#普通随机打乱，不限制身份 / 摄像头 / 数据集，标准分类任务用，不能用于三元组损失训练。
        sampler = RandomSampler(data_source)

    return sampler
