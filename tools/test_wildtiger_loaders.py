from __future__ import absolute_import, print_function

import argparse
import csv
import json
import os.path as osp
import re
import sys


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torchreid.data import ImageDataManager


def parse_args():
    parser = argparse.ArgumentParser(
        description='Load and inspect the first WildTiger data batches'
    )
    parser.add_argument(
        '--root',
        default=osp.join(PROJECT_ROOT, 'reid-data'),
        help='dataset root containing the Wild_Tiger directory'
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


def read_csv_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def check_test_csvs(dataset_dir):
    query_rows = read_csv_rows(osp.join(dataset_dir, 'query.csv'))
    gallery_rows = read_csv_rows(osp.join(dataset_dir, 'gallery.csv'))
    rows_by_pid = {}
    for split, rows in [('query', query_rows), ('gallery', gallery_rows)]:
        for row in rows:
            rows_by_pid.setdefault(row['pid'], []).append((split, row))

    for pid, rows in rows_by_pid.items():
        query_type = {row['query'] for _, row in rows}
        if len(query_type) != 1:
            raise AssertionError('Inconsistent query type for PID {}'.format(pid))
        query_type = next(iter(query_type))
        if query_type == 'sing':
            query_cams = {
                int(row['camid']) for split, row in rows if split == 'query'
            }
            gallery_cams = {
                int(row['camid']) for split, row in rows if split == 'gallery'
            }
            if query_cams != {0} or gallery_cams != {1}:
                raise AssertionError(
                    'Invalid single-video pseudo camids for PID {}'.format(pid)
                )
        elif query_type == 'multi':
            video_to_camid = {}
            for _, row in rows:
                match = re.search(r'(video_\d+)', row['image'], re.IGNORECASE)
                if match is None:
                    raise AssertionError('Missing video name in {}'.format(row['image']))
                video = match.group(1).lower()
                camid = int(row['camid'])
                if video in video_to_camid and video_to_camid[video] != camid:
                    raise AssertionError(
                        'Inconsistent camid for {} of PID {}'.format(video, pid)
                    )
                video_to_camid[video] = camid
            expected = set(range(1, len(video_to_camid) + 1))
            if set(video_to_camid.values()) != expected:
                raise AssertionError(
                    'Multi-video camids are not contiguous for PID {}'.format(pid)
                )
        else:
            raise AssertionError('Invalid query type: {}'.format(query_type))


def main():
    args = parse_args()
    datamanager = ImageDataManager(
        root=args.root,
        sources='wildtiger',
        targets='wildtiger',
        height=args.height,
        width=args.width,
        batch_size_train=args.batch_size,
        batch_size_test=8,
        workers=args.workers,
        use_gpu=False
    )

    if 'wildtiger' not in datamanager.val_loader:
        raise RuntimeError('WildTiger validation loader was not created')

    valset = datamanager.val_loader['wildtiger'].dataset
    manifest_path = osp.join(args.root, 'Wild_Tiger', 'split.json')
    with open(manifest_path, 'r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    check_test_csvs(osp.join(args.root, 'Wild_Tiger'))
    expected_val_images = sum(
        item['images'] for item in manifest['splits']['val']
    )
    expected_gallery_images = sum(
        item['gallery_images'] for item in manifest['splits']['test']
    )
    test_multi_ids = sum(
        len(item['videos']) > 1 for item in manifest['splits']['test']
    )
    test_single_ids = sum(
        len(item['videos']) == 1 for item in manifest['splits']['test']
    )
    train_pids = {item[1] for item in valset.train}
    val_pids = {item[1] for item in valset.val}
    if len(valset) != expected_val_images:
        raise AssertionError(
            'Expected {} validation images, got {}'.format(
                expected_val_images, len(valset)
            )
        )
    if val_pids != train_pids:
        raise AssertionError('Validation PIDs do not match training PIDs')
    if val_pids != set(range(valset.num_train_pids)):
        raise AssertionError('Validation PIDs are not contiguous train labels')

    test_loaders = datamanager.test_loader['wildtiger']
    queryset = test_loaders['query'].dataset
    galleryset = test_loaders['gallery'].dataset
    query_counts = {}
    for item in queryset.query:
        query_counts[item[1]] = query_counts.get(item[1], 0) + 1
    if len(queryset) != 160:
        raise AssertionError(
            'Expected 160 query images, got {}'.format(len(queryset))
        )
    if len(galleryset) != expected_gallery_images:
        raise AssertionError(
            'Expected {} gallery images, got {}'.format(
                expected_gallery_images, len(galleryset)
            )
        )
    if len(query_counts) != 80 or set(query_counts.values()) != {2}:
        raise AssertionError('Each test identity must have 2 query images')
    if (test_multi_ids, test_single_ids) != (30, 50):
        raise AssertionError(
            'Expected 30 multi-video and 50 single-video test identities, '
            'got {} and {}'.format(test_multi_ids, test_single_ids)
        )

    print('\nFirst batch tensor shapes')
    print_first_batch('Train', datamanager.train_loader)
    print_first_batch('Validation', datamanager.val_loader['wildtiger'])
    print_first_batch('Test Query', test_loaders['query'])
    print_first_batch('Test Gallery', test_loaders['gallery'])


if __name__ == '__main__':
    main()
