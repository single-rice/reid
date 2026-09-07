from __future__ import division, print_function, absolute_import
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from torchreid import metrics
from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results
)
from torchreid.losses import DeepSupervision


def evaluate_rank_by_query_group(
    distmat,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    q_groups,
    use_metric_cuhk03=False
):
    """Evaluates query groups against one shared, complete gallery."""
    q_groups = np.asarray(q_groups)
    if q_groups.shape[0] != distmat.shape[0]:
        raise ValueError(
            'Expected one evaluation group per query, but got {} groups for '
            '{} queries'.format(q_groups.shape[0], distmat.shape[0])
        )
    unknown_groups = set(np.unique(q_groups)) - {'sing', 'multi'}
    if unknown_groups:
        raise ValueError(
            'Unknown evaluation query groups: {}'.format(
                ', '.join(sorted(unknown_groups))
            )
        )

    results = OrderedDict()
    for result_name, group_value in (
        ('single_video', 'sing'),
        ('cross_video', 'multi')
    ):
        mask = q_groups == group_value
        if not np.any(mask):
            continue
        cmc, mAP = metrics.evaluate_rank(
            distmat[mask],
            q_pids[mask],
            g_pids,
            q_camids[mask],
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )
        results[result_name] = {
            'num_queries': int(mask.sum()),
            'num_entities': int(np.unique(q_pids[mask]).size),
            'cmc': cmc,
            'mAP': mAP
        }
    return results

#负责训练循环、测试流程、特征提取、保存模型
class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, use_gpu=True):
        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.epoch = 0
        self.test_results = OrderedDict()

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, rank1, save_dir, is_best=False):
        names = self.get_model_names()

        for name in names:
            save_checkpoint(
                {
                    'state_dict': self._models[name].state_dict(),
                    'epoch': epoch + 1,
                    'rank1': rank1,
                    'optimizer': self._optims[name].state_dict(),
                    'scheduler': self._scheds[name].state_dict()
                },
                osp.join(save_dir, name),
                is_best=is_best
            )

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[-1]['lr']

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def run(
        self,
        save_dir='log',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        test_only=False,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
                统一的模型训练与评测流水线引擎

            参数说明：
                save_dir (str): 模型、日志、可视化结果保存目录
                max_epoch (int): 完整训练总轮数
                start_epoch (int, 可选): 训练起始轮次，用于断点续训，默认0
                print_freq (int, 可选): 每隔多少batch打印一次训练loss/指标，默认10
                fixbase_epoch (int, 可选): 两阶段迁移学习冻结骨干轮数；前fixbase_epoch轮仅训练open_layers指定新增层，骨干完全冻结；该轮次计入总max_epoch，默认0（不冻结骨干，全程全开训练）
                open_layers (str / list, 可选): 解冻训练的新增层名称，如classifier、attn、neck，配合fixbase_epoch使用
                start_eval (int, 可选): 从第几轮开始执行测试评估，默认0（每eval_freq轮都会测）
                eval_freq (int, 可选): 每隔多少轮执行一次完整测试；默认-1，仅在全部训练结束后评测一次
                test_only (bool, 可选): 仅执行测试，不进行训练；加载已有权重直接跑评测，默认False
                dist_metric (str, 可选): 计算query与gallery特征距离的度量方式，默认euclidean（欧氏距离），可选cosine（余弦距离）
                normalize_feature (bool, 可选): 提取特征后先做L2归一化再计算距离；开启后等价使用余弦距离，搭配cosine度量常用，默认False
                visrank (bool, 可选): 可视化检索排序结果；建议test_only=True时开启，匹配图片保存至 save_dir/visrank_数据集名，默认False
                visrank_topk (int, 可选): 可视化每个query前Top-k匹配图，默认10
                use_metric_cuhk03 (bool, 可选): CUHK03数据集单图库评测协议；使用经典划分时必须开启，默认False
                ranks (list, 可选): CMC曲线计算的Rank指标，默认[1,5,10,20]（Rank1/Rank5/Rank10/Rank20）
                rerank (bool, 可选): 开启行人重排序算法（CVPR2017文章），大幅提升mAP指标；仅test_only=True时生效，默认False

        """

        if visrank and not test_only:
            raise ValueError(
                'visrank can be set to True only if test_only=True'
            )

        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        print('=> Start training')

        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.train(
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers
            )

            if (self.epoch + 1) >= start_eval \
               and eval_freq > 0 \
               and (self.epoch+1) % eval_freq == 0 \
               and (self.epoch + 1) != self.max_epoch:
                rank1 = self.test(
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks
                )
                self.save_model(self.epoch, rank1, save_dir)

        if self.max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks
            )
            self.save_model(self.epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()

    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_model_mode('train')

        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )

        self.num_batches = len(self.train_loader)
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta_str,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            if self.writer is not None:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                self.writer.add_scalar(
                    'Train/lr', self.get_current_lr(), n_iter
                )

            end = time.time()

        self.update_lr()

    def forward_backward(self, data):
        raise NotImplementedError

    def test(
        self,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        self.set_model_mode('eval')
        targets = list(self.test_loader.keys())

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = self.test_loader[name]['query']
            gallery_loader = self.test_loader[name]['gallery']
            rank1, mAP = self._evaluate(
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )

            if self.writer is not None:
                self.writer.add_scalar(f'Test/{name}/rank1', rank1, self.epoch)
                self.writer.add_scalar(f'Test/{name}/mAP', mAP, self.epoch)
                for group_name, result in self.test_results[name].items():
                    if group_name == 'overall':
                        continue
                    self.writer.add_scalar(
                        f'Test/{name}/{group_name}/rank1',
                        result['cmc'][0],
                        self.epoch
                    )
                    self.writer.add_scalar(
                        f'Test/{name}/{group_name}/mAP',
                        result['mAP'],
                        self.epoch
                    )

        return rank1

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False
    ):
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            f_, pids_, camids_, groups_ = [], [], [], []
            has_groups = None
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self.extract_features(imgs)
                batch_time.update(time.time() - end)
                features = features.cpu()
                f_.append(features)
                pids_.extend(pids.tolist())
                camids_.extend(camids.tolist())
                batch_groups = data.get('eval_group')
                batch_has_groups = batch_groups is not None
                if has_groups is None:
                    has_groups = batch_has_groups
                elif has_groups != batch_has_groups:
                    raise RuntimeError(
                        'Evaluation group metadata is missing from part of '
                        'the data loader'
                    )
                if batch_has_groups:
                    groups_.extend(batch_groups)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            groups_ = np.asarray(groups_) if has_groups else None
            return f_, pids_, camids_, groups_

        print('Extracting features from query set ...')
        qf, q_pids, q_camids, q_groups = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids, _ = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(dist_metric)
        )
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

        dataset_results = OrderedDict()
        dataset_results['overall'] = {
            'num_queries': int(len(q_pids)),
            'num_entities': int(np.unique(q_pids).size),
            'cmc': cmc,
            'mAP': mAP
        }
        if q_groups is not None:
            group_results = evaluate_rank_by_query_group(
                distmat,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                q_groups,
                use_metric_cuhk03=use_metric_cuhk03
            )
            dataset_results.update(group_results)
            for group_name, result in group_results.items():
                print('** {} Results ({} queries, {} entities) **'.format(
                    group_name.replace('_', '-').title(),
                    result['num_queries'],
                    result['num_entities']
                ))
                print('mAP: {:.1%}'.format(result['mAP']))
                print('CMC curve')
                for r in ranks:
                    print('Rank-{:<3}: {:.1%}'.format(
                        r, result['cmc'][r - 1]
                    ))
        self.test_results[dataset_name] = dataset_results

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        return cmc[0], mAP

    
    def compute_loss(self, criterion, outputs, targets):
        # 判断模型输出是否是元组/列表（多分支、深度监督输出）
        if isinstance(outputs, (tuple, list)):
        # 启用深度监督损失计算
            loss = DeepSupervision(criterion, outputs, targets)
        else:
        # 单路输出，直接用损失函数计算损失
            loss = criterion(outputs, targets)
        return loss

    def extract_features(self, input):
        return self.model(input)

    def parse_data_for_train(self, data):
        imgs = data['img']
        pids = data['pid']
        return imgs, pids

    def parse_data_for_eval(self, data):
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        return imgs, pids, camids

    def two_stepped_transfer_learning(
        self, epoch, fixbase_epoch, open_layers, model=None
    ):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """
        model = self.model if model is None else model
        if model is None:
            return

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(model, open_layers)
        else:
            open_all_layers(model)
