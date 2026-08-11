from __future__ import print_function, absolute_import

import argparse
import csv
import os
import os.path as osp
import re

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(
    r'epoch: \[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*'
    r'loss ([0-9.]+) \(([0-9.]+)\).*'
    r'acc ([0-9.]+) \(([0-9.]+)\)'
)
TRIPLET_TRAIN_RE = re.compile(
    r'epoch: \[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*'
    r'loss_t ([0-9.]+) \(([0-9.]+)\).*'
    r'loss_x ([0-9.]+) \(([0-9.]+)\).*'
    r'acc ([0-9.]+) \(([0-9.]+)\)'
)
MAP_RE = re.compile(r'mAP: ([0-9.]+)%')
RANK1_RE = re.compile(r'Rank-1\s*: ([0-9.]+)%')
RANK5_RE = re.compile(r'Rank-5\s*: ([0-9.]+)%')
RANK10_RE = re.compile(r'Rank-10\s*: ([0-9.]+)%')
RANK20_RE = re.compile(r'Rank-20\s*: ([0-9.]+)%')


def parse_logs(log_paths):
    train_by_epoch = {}
    eval_by_epoch = {}
    last_train_epoch = None
    pending_eval = None
    pending_map = None

    for log_path in log_paths:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                train_match = TRAIN_RE.search(line)
                if train_match:
                    epoch = int(train_match.group(1))
                    last_train_epoch = epoch
                    train_by_epoch[epoch] = {
                        'epoch': epoch,
                        'train_loss': float(train_match.group(6)),
                        'loss_t': None,
                        'loss_x': None,
                        'train_acc': float(train_match.group(8)),
                    }
                    continue

                triplet_train_match = TRIPLET_TRAIN_RE.search(line)
                if triplet_train_match:
                    epoch = int(triplet_train_match.group(1))
                    last_train_epoch = epoch
                    loss_t = float(triplet_train_match.group(6))
                    loss_x = float(triplet_train_match.group(8))
                    train_by_epoch[epoch] = {
                        'epoch': epoch,
                        'train_loss': loss_t + loss_x,
                        'loss_t': loss_t,
                        'loss_x': loss_x,
                        'train_acc': float(triplet_train_match.group(10)),
                    }
                    continue

                map_match = MAP_RE.search(line)
                if map_match:
                    pending_map = float(map_match.group(1))
                    pending_eval = {'epoch': last_train_epoch, 'mAP': pending_map}
                    continue

                rank1_match = RANK1_RE.search(line)
                if rank1_match and pending_eval is not None:
                    pending_eval['rank1'] = float(rank1_match.group(1))
                    continue

                rank5_match = RANK5_RE.search(line)
                if rank5_match and pending_eval is not None:
                    pending_eval['rank5'] = float(rank5_match.group(1))
                    continue

                rank10_match = RANK10_RE.search(line)
                if rank10_match and pending_eval is not None:
                    pending_eval['rank10'] = float(rank10_match.group(1))
                    continue

                rank20_match = RANK20_RE.search(line)
                if rank20_match and pending_eval is not None:
                    pending_eval['rank20'] = float(rank20_match.group(1))
                    if pending_eval['epoch'] is not None:
                        eval_by_epoch[pending_eval['epoch']] = pending_eval
                    pending_eval = None
                    pending_map = None

    rows = []
    epochs = sorted(train_by_epoch)
    for epoch in epochs:
        row = dict(train_by_epoch[epoch])
        row.update(
            {
                'test_loss': None,
                'mAP': None,
                'rank1': None,
                'rank5': None,
                'rank10': None,
                'rank20': None,
            }
        )
        if epoch in eval_by_epoch:
            row.update(eval_by_epoch[epoch])
        rows.append(row)

    return rows


def save_csv(rows, out_csv):
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'epoch', 'train_loss', 'train_acc', 'test_loss', 'mAP',
                'rank1', 'rank5', 'rank10', 'rank20', 'loss_t', 'loss_x'
            ]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot(rows, out_png, title):
    epochs = [row['epoch'] for row in rows]
    train_loss = [row['train_loss'] for row in rows]
    loss_t = [row.get('loss_t') for row in rows]
    loss_x = [row.get('loss_x') for row in rows]
    train_acc = [row['train_acc'] for row in rows]
    maps = [row.get('mAP') for row in rows]
    rank1 = [row.get('rank1') for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    max_epoch = max(epochs)
    xticks = [1] + list(range(10, max_epoch + 1, 10))

    axes[0].plot(epochs, train_loss, linewidth=1.8, label='Train loss')
    if any(value is not None for value in loss_t):
        axes[0].plot(epochs, loss_t, linewidth=1.2, label='Triplet loss')
    if any(value is not None for value in loss_x):
        axes[0].plot(epochs, loss_x, linewidth=1.2, label='Cross entropy loss')
    axes[0].set_title('{} Loss'.format(title))
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross entropy loss')
    axes[0].set_xticks(xticks if len(epochs) > 15 else epochs)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].text(
        0.02,
        0.03,
        'Test loss: N/A for standard ReID evaluation\n'
        '(query/gallery IDs are unseen by the classifier)',
        transform=axes[0].transAxes,
        fontsize=8,
        va='bottom'
    )

    axes[1].plot(epochs, maps, marker='o', linewidth=1.8, label='mAP')
    axes[1].plot(epochs, rank1, marker='s', linewidth=1.8, label='Rank-1')
    axes[1].plot(epochs, train_acc, linewidth=1.2, alpha=0.65, label='Train acc')
    axes[1].set_title('{} Retrieval Metrics'.format(title))
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Percent')
    axes[1].set_xticks(xticks if len(epochs) > 15 else epochs)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, nargs='+')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--title', default='TigerMini')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = parse_logs(args.log)
    if not rows:
        raise RuntimeError('No metric rows parsed from {}'.format(args.log))

    out_csv = osp.join(args.out_dir, 'tigermini_metrics.csv')
    out_png = osp.join(args.out_dir, 'tigermini_metrics.png')
    save_csv(rows, out_csv)
    plot(rows, out_png, args.title)
    print('Saved {}'.format(out_csv))
    print('Saved {}'.format(out_png))


if __name__ == '__main__':
    main()
