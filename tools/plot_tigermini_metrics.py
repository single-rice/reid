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
MAP_RE = re.compile(r'mAP: ([0-9.]+)%')
RANK1_RE = re.compile(r'Rank-1\s*: ([0-9.]+)%')


def parse_log(log_path):
    train_by_epoch = {}
    eval_rows = []
    pending_map = None

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_match = TRAIN_RE.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                train_by_epoch[epoch] = {
                    'epoch': epoch,
                    'train_loss': float(train_match.group(6)),
                    'train_acc': float(train_match.group(8)),
                }
                continue

            map_match = MAP_RE.search(line)
            if map_match:
                pending_map = float(map_match.group(1))
                continue

            rank1_match = RANK1_RE.search(line)
            if rank1_match and pending_map is not None:
                eval_rows.append(
                    {
                        'epoch': len(eval_rows) + 1,
                        'mAP': pending_map,
                        'rank1': float(rank1_match.group(1)),
                    }
                )
                pending_map = None

    rows = []
    epochs = sorted(train_by_epoch)
    for idx, epoch in enumerate(epochs):
        row = dict(train_by_epoch[epoch])
        if idx < len(eval_rows):
            row.update(
                {
                    'mAP': eval_rows[idx]['mAP'],
                    'rank1': eval_rows[idx]['rank1'],
                    'test_loss': None,
                }
            )
        rows.append(row)

    return rows


def save_csv(rows, out_csv):
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'epoch', 'train_loss', 'train_acc', 'test_loss', 'mAP',
                'rank1'
            ]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot(rows, out_png):
    epochs = [row['epoch'] for row in rows]
    train_loss = [row['train_loss'] for row in rows]
    train_acc = [row['train_acc'] for row in rows]
    maps = [row.get('mAP') for row in rows]
    rank1 = [row.get('rank1') for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)

    axes[0].plot(epochs, train_loss, marker='o', label='Train loss')
    axes[0].set_title('TigerMini Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross entropy loss')
    axes[0].set_xticks(epochs)
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

    axes[1].plot(epochs, maps, marker='o', label='mAP')
    axes[1].plot(epochs, rank1, marker='s', label='Rank-1')
    axes[1].plot(epochs, train_acc, marker='^', label='Train acc')
    axes[1].set_title('TigerMini Retrieval Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Percent')
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = parse_log(args.log)
    if not rows:
        raise RuntimeError('No metric rows parsed from {}'.format(args.log))

    out_csv = osp.join(args.out_dir, 'tigermini_metrics.csv')
    out_png = osp.join(args.out_dir, 'tigermini_metrics.png')
    save_csv(rows, out_csv)
    plot(rows, out_png)
    print('Saved {}'.format(out_csv))
    print('Saved {}'.format(out_png))


if __name__ == '__main__':
    main()
