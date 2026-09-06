from __future__ import absolute_import, print_function

import argparse
import os.path as osp
import sys


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torchreid.data import ImageDataManager


def parse_args():
    parser = argparse.ArgumentParser(
        description='Load and inspect the first ATRW data batches'
    )
    parser.add_argument(
        '--root',
        default=osp.join(PROJECT_ROOT, 'reid-data'),
        help='dataset root containing the ATRW directory'
    )
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=256)
    return parser.parse_args()


def print_first_batch(name, loader):
    batch = next(iter(loader))
    print('{}:'.format(name))
    print('  img shape   = {}'.format(tuple(batch['img'].shape)))
    print('  pid shape   = {}'.format(tuple(batch['pid'].shape)))
    print('  camid shape = {}'.format(tuple(batch['camid'].shape)))


def main():
    args = parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    print('Python: {}'.format(sys.executable), flush=True)
    print('ATRW root: {}'.format(osp.join(args.root, 'ATRW')), flush=True)
    print('Building ATRW data loaders ...', flush=True)
    datamanager = ImageDataManager(
        root=args.root,
        sources='atrw',
        targets='atrw',
        height=args.height,
        width=args.width,
        batch_size_train=args.batch_size,
        batch_size_test=8,
        workers=args.workers,
        use_gpu=False
    )

    test_loaders = datamanager.test_loader['atrw']
    print('\nFirst batch tensor shapes')
    print_first_batch('Train', datamanager.train_loader)
    print_first_batch('Test Query', test_loaders['query'])
    print_first_batch('Test Gallery', test_loaders['gallery'])


if __name__ == '__main__':
    main()
