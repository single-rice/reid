from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 让脚本在 Windows/Linux 上从任意当前目录运行时，都能找到项目根目录里的 torchreid。
THIS_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use('Agg')  # 服务器通常没有图形界面，Agg 可以直接把图保存成 png。
import matplotlib.pyplot as plt

import torchreid
from default_config import get_default_config, imagedata_kwargs, videodata_kwargs
from torchreid.utils import check_isfile, load_pretrained_weights, mkdir_if_missing


def parse_args():
    parser = argparse.ArgumentParser(
        description='使用 UMAP/t-SNE 可视化 ReID 特征空间',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='训练时使用的 yaml 配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='要加载的模型 checkpoint，例如 log/.../model.pth.tar-60'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='log/feature_vis',
        help='可视化图片和 csv 的保存目录'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='',
        help='要可视化的测试集名称；留空则使用配置里的第一个 target'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='gallery',
        choices=['gallery', 'query', 'query_gallery'],
        help='使用 gallery、query，还是 query+gallery 一起画图'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='umap',
        choices=['umap', 'tsne'],
        help='二维降维方法'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='最多可视化多少张图；0 表示使用全部样本'
    )
    parser.add_argument(
        '--normalize-feature',
        action='store_true',
        help='降维前是否对 ReID 特征做 L2 归一化'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='强制使用 CPU，即使机器上有 CUDA'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='随机种子，用于 UMAP/t-SNE 和抽样'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='输出图片 dpi'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='临时覆盖配置项，例如 data.root / test.batch_size / data.workers'
    )
    return parser.parse_args()


def build_datamanager(cfg):
    """按 main.py 的方式构建 datamanager，保证数据读取逻辑和训练/测试一致。"""
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_model(cfg, datamanager):
    """按 main.py 的方式构建模型，然后加载 checkpoint 权重。"""
    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=False,  # 这里马上会加载 checkpoint，不再额外下载/加载 ImageNet 预训练权重。
        use_gpu=cfg.use_gpu
    )

    if not check_isfile(cfg.model.load_weights):
        raise FileNotFoundError('checkpoint 不存在: {}'.format(cfg.model.load_weights))

    # load_pretrained_weights 会自动处理 checkpoint 里的 state_dict 和 module. 前缀。
    load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()
    model.eval()
    return model


def unpack_eval_batch(data):
    """从 torchreid 的 batch 字典中取出图片、身份 ID、摄像头 ID 和图片路径。"""
    imgs = data['img']
    pids = data['pid']
    camids = data['camid']
    impaths = data.get('impath', [''] * len(pids))
    return imgs, pids, camids, impaths


@torch.no_grad()
def extract_features(
    model,
    loader,
    split_name,
    use_gpu,
    normalize_feature=False,
    max_samples=0
):
    """遍历 query/gallery loader，得到每张图片的特征向量和标签信息。"""
    features, pids, camids, impaths, splits = [], [], [], [], []

    for batch_idx, data in enumerate(loader):
        imgs, pid, camid, impath = unpack_eval_batch(data)
        if use_gpu:
            imgs = imgs.cuda()

        output = model(imgs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if normalize_feature:
            output = F.normalize(output, p=2, dim=1)

        features.append(output.cpu())
        pids.extend(pid.cpu().numpy().tolist())
        camids.extend(camid.cpu().numpy().tolist())
        impaths.extend(list(impath))
        splits.extend([split_name] * len(pid))

        if (batch_idx + 1) % 20 == 0:
            print('  {}: 已处理 {} 个 batch'.format(split_name, batch_idx + 1))

        # 做快速预览时，不必把整个 gallery/query 都跑完。
        if max_samples > 0 and len(pids) >= max_samples:
            break

    features = torch.cat(features, dim=0).numpy()
    if max_samples > 0 and len(pids) > max_samples:
        features = features[:max_samples]
        pids = pids[:max_samples]
        camids = camids[:max_samples]
        impaths = impaths[:max_samples]
        splits = splits[:max_samples]

    return features, np.asarray(pids), np.asarray(camids), impaths, splits


def collect_split_features(datamanager, model, target, split, cfg, max_samples=0):
    """根据参数选择 query/gallery，并把不同 split 的特征拼起来。"""
    loaders = datamanager.test_loader[target]
    parts = []
    remaining = max_samples

    if split in ['query', 'query_gallery']:
        print('Extracting features from query set ...')
        limit = remaining if remaining > 0 else 0
        parts.append(
            extract_features(
                model,
                loaders['query'],
                'query',
                cfg.use_gpu,
                normalize_feature=cfg.test.normalize_feature,
                max_samples=limit
            )
        )
        if remaining > 0:
            remaining = max(0, remaining - len(parts[-1][1]))

    if split in ['gallery', 'query_gallery'] and remaining != 0:
        print('Extracting features from gallery set ...')
        limit = remaining if remaining > 0 else 0
        parts.append(
            extract_features(
                model,
                loaders['gallery'],
                'gallery',
                cfg.use_gpu,
                normalize_feature=cfg.test.normalize_feature,
                max_samples=limit
            )
        )

    features = np.concatenate([item[0] for item in parts], axis=0)
    pids = np.concatenate([item[1] for item in parts], axis=0)
    camids = np.concatenate([item[2] for item in parts], axis=0)
    impaths = sum([item[3] for item in parts], [])
    splits = sum([item[4] for item in parts], [])
    print('Done, obtained {}-by-{} feature matrix'.format(features.shape[0], features.shape[1]))
    return features, pids, camids, impaths, splits


def sample_features(features, pids, camids, impaths, splits, max_samples, seed):
    """样本很多时可以随机抽一部分，避免 UMAP/t-SNE 太慢或图太挤。"""
    if max_samples <= 0 or len(pids) <= max_samples:
        return features, pids, camids, impaths, splits

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(pids), size=max_samples, replace=False)
    indices = np.sort(indices)
    return (
        features[indices],
        pids[indices],
        camids[indices],
        [impaths[i] for i in indices],
        [splits[i] for i in indices]
    )


def check_reducer_dependency(method):
    """提前检查降维依赖，避免特征都提完了才发现缺包。"""
    if method == 'umap':
        try:
            import umap  # noqa: F401
        except ImportError:
            raise ImportError(
                'Missing dependency: umap-learn. Please install it with: pip install umap-learn'
            )
    else:
        try:
            from sklearn.manifold import TSNE  # noqa: F401
        except ImportError:
            raise ImportError('Missing dependency: scikit-learn. Cannot use t-SNE.')


def reduce_to_2d(features, method, seed):
    """把高维 ReID 特征降到二维，方便画散点图。"""
    if len(features) < 2:
        raise ValueError('至少需要 2 个样本才能做二维可视化')

    if method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError(
                'Missing dependency: umap-learn. Please install it with: pip install umap-learn'
            )

        n_neighbors = min(15, max(2, len(features) - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='euclidean',
            random_state=seed
        )
        return reducer.fit_transform(features)

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError('Missing dependency: scikit-learn. Cannot use t-SNE.')

    perplexity = min(30, max(1, (len(features) - 1) // 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=seed
    )
    return reducer.fit_transform(features)


def plot_embedding(embedding, pids, splits, title, save_path, dpi):
    """按 person ID / tiger ID 给点上色，query/gallery 用不同 marker 区分。"""
    unique_pids = np.unique(pids)
    pid_to_color = {pid: idx for idx, pid in enumerate(unique_pids)}
    color_ids = np.asarray([pid_to_color[pid] for pid in pids])

    if len(unique_pids) <= 20:
        cmap = plt.get_cmap('tab20', len(unique_pids))
    else:
        cmap = plt.get_cmap('hsv', len(unique_pids))

    plt.figure(figsize=(10, 8))
    marker_map = {'query': 'o', 'gallery': '^'}
    for split_name in sorted(set(splits)):
        mask = np.asarray([item == split_name for item in splits])
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color_ids[mask],
            cmap=cmap,
            s=16,
            alpha=0.85,
            marker=marker_map.get(split_name, 'o'),
            linewidths=0,
            label=split_name
        )

    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.35)
    plt.legend(loc='best')

    cbar = plt.colorbar()
    cbar.set_label('person ID / tiger ID')
    if len(unique_pids) <= 30:
        cbar.set_ticks(np.arange(len(unique_pids)))
        cbar.set_ticklabels([str(pid) for pid in unique_pids])

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def save_metadata_csv(csv_path, embedding, pids, camids, impaths, splits):
    """保存每个点对应的身份、摄像头和图片路径，方便之后排查某个簇。"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'pid', 'camid', 'split', 'impath'])
        for xy, pid, camid, split, impath in zip(embedding, pids, camids, splits, impaths):
            writer.writerow([xy[0], xy[1], int(pid), int(camid), split, impath])


def main():
    args = parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available() and not args.no_cuda
    cfg.merge_from_file(args.config_file)
    cfg.model.load_weights = args.checkpoint
    cfg.test.normalize_feature = args.normalize_feature
    if args.opts:
        cfg.merge_from_list(args.opts)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    check_reducer_dependency(args.method)

    datamanager = build_datamanager(cfg)
    target = args.target if args.target else cfg.data.targets[0]
    if target not in datamanager.test_loader:
        raise KeyError(
            'target={} 不在 datamanager.test_loader 中，可选: {}'.format(
                target, list(datamanager.test_loader.keys())
            )
        )

    model = build_model(cfg, datamanager)
    features, pids, camids, impaths, splits = collect_split_features(
        datamanager, model, target, args.split, cfg, max_samples=args.max_samples
    )
    features, pids, camids, impaths, splits = sample_features(
        features, pids, camids, impaths, splits, args.max_samples, args.seed
    )

    print('Running {} on {} samples ...'.format(args.method.upper(), len(pids)))
    embedding = reduce_to_2d(features, args.method, args.seed)

    mkdir_if_missing(args.output_dir)
    checkpoint_name = osp.basename(args.checkpoint)
    stem = '{}_{}_{}_{}'.format(args.method, target, args.split, checkpoint_name)
    stem = stem.replace('/', '_').replace('\\', '_').replace(':', '_')
    png_path = osp.join(args.output_dir, stem + '.png')
    csv_path = osp.join(args.output_dir, stem + '.csv')

    title = '{} feature space ({}, {}, n={})'.format(
        args.method.upper(), target, args.split, len(pids)
    )
    plot_embedding(embedding, pids, splits, title, png_path, args.dpi)
    save_metadata_csv(csv_path, embedding, pids, camids, impaths, splits)

    print('可视化图片已保存到: {}'.format(osp.abspath(png_path)))
    print('点位元数据已保存到: {}'.format(osp.abspath(csv_path)))


if __name__ == '__main__':
    main()
