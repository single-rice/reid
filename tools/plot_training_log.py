from __future__ import absolute_import, print_function

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


LINE_PATTERN = re.compile(
    r'epoch:\s*\[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?'
    r'loss\s+[\d.eE+-]+\s+\(([\d.eE+-]+)\).*?'
    r'acc\s+[\d.eE+-]+\s+\(([\d.eE+-]+)\)'
)
CURRENT_BATCH_PATTERN = re.compile(
    r'epoch:\s*\[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?'
    r'loss\s+([\d.eE+-]+)\s+\([\d.eE+-]+\).*?'
    r'acc\s+([\d.eE+-]+)\s+\([\d.eE+-]+\)'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot epoch loss and accuracy from a torchreid log'
    )
    parser.add_argument('log_file')
    parser.add_argument('--output', default='')
    parser.add_argument(
        '--by-batch', action='store_true',
        help='plot current batch metrics against global batch step'
    )
    return parser.parse_args()


def parse_epoch_metrics(log_file):
    # Later printed batches replace earlier ones, leaving the latest cumulative
    # average available for each epoch.
    epochs = {}
    with log_file.open('r', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            match = LINE_PATTERN.search(line)
            if match:
                epoch, _, batch, total_batches, loss, accuracy = match.groups()
                epochs[int(epoch)] = {
                    'epoch': int(epoch),
                    'last_logged_batch': int(batch),
                    'total_batches': int(total_batches),
                    'loss': float(loss),
                    'accuracy': float(accuracy),
                }
    if not epochs:
        raise RuntimeError('No epoch metrics found in {}'.format(log_file))
    return [epochs[epoch] for epoch in sorted(epochs)]


def parse_batch_metrics(log_file):
    metrics = []
    with log_file.open('r', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            match = CURRENT_BATCH_PATTERN.search(line)
            if not match:
                continue
            epoch, _, batch, total_batches, loss, accuracy = match.groups()
            epoch, batch, total_batches = map(
                int, (epoch, batch, total_batches)
            )
            metrics.append({
                'global_batch': (epoch - 1) * total_batches + batch,
                'epoch': epoch,
                'batch': batch,
                'total_batches': total_batches,
                'loss': float(loss),
                'accuracy': float(accuracy),
            })
    if not metrics:
        raise RuntimeError('No batch metrics found in {}'.format(log_file))
    return metrics


def main():
    args = parse_args()
    log_file = Path(args.log_file).resolve()
    suffix = '_batch_loss_accuracy.png' if args.by_batch else '_loss_accuracy.png'
    output = (
        Path(args.output).resolve() if args.output
        else log_file.with_name(log_file.name + suffix)
    )
    csv_file = output.with_suffix('.csv')
    metrics = (
        parse_batch_metrics(log_file) if args.by_batch
        else parse_epoch_metrics(log_file)
    )

    with csv_file.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    x_key = 'global_batch' if args.by_batch else 'epoch'
    x_values = [item[x_key] for item in metrics]
    losses = [item['loss'] for item in metrics]
    accuracies = [item['accuracy'] for item in metrics]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x_values, losses, color='#d95f02', linewidth=1)
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('WildTiger Training Curves')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x_values, accuracies, color='#1b9e77', linewidth=1)
    axes[1].set_xlabel('Global Batch Step' if args.by_batch else 'Epoch')
    axes[1].set_ylabel('Training Accuracy (%)')
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output), dpi=200, bbox_inches='tight')
    plt.close(fig)
    unit = 'logged batches' if args.by_batch else 'epochs'
    print('Parsed {} {}'.format(len(metrics), unit))
    print('Saved plot: {}'.format(output))
    print('Saved CSV: {}'.format(csv_file))


if __name__ == '__main__':
    main()
