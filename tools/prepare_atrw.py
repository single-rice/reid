"""Prepare ATRW train/query/gallery splits used by torchreid.

The official training set is kept intact. For every identity in the official
test annotations, two images are selected deterministically for query and all
remaining images are placed in gallery.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import os.path as osp
import random
import shutil
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare the ATRW dataset')
    parser.add_argument('--root', default=osp.join('reid-data', 'ATRW'))
    parser.add_argument('--query-per-id', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_train_annotations(path):
    records = []
    with open(path, 'r') as csv_file:
        for row in csv.reader(csv_file):
            if len(row) >= 2:
                records.append((int(row[0]), row[1].strip()))
    return records


def read_test_annotations(path):
    with open(path, 'r') as json_file:
        annotations = json.load(json_file)
    return [
        {
            'pid': int(item['entityid']),
            'image': '{:06d}.jpg'.format(int(item['imgid'])),
            'query_type': item['query'],
            # The first two frame fields identify the source clip; the last
            # field is the frame number inside that clip.
            'video': tuple(item['frame'][:2])
        }
        for item in annotations
    ]


def write_split(root, name, records, source_dir, overwrite, test_split=False):
    output_dir = osp.join(root, name)
    output_csv = osp.join(root, name + '.csv')
    if osp.exists(output_dir):
        if not overwrite:
            raise RuntimeError('{} already exists; pass --overwrite'.format(output_dir))
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['pid', 'image', 'camid', 'query'] if test_split else
                        ['pid', 'image'])
        for record in records:
            pid, image_name = record[:2]
            source = osp.join(source_dir, image_name)
            if not osp.isfile(source):
                raise RuntimeError('Annotated image not found: {}'.format(source))
            shutil.copy2(source, osp.join(output_dir, image_name))
            writer.writerow(record)


def main():
    args = parse_args()
    root = osp.abspath(osp.expanduser(args.root))
    if args.query_per_id < 1:
        raise ValueError('--query-per-id must be positive')

    train_source = osp.join(root, 'atrw_reid_train', 'train')
    test_source = osp.join(root, 'atrw_reid_test', 'test')
    train_annotations = osp.join(root, 'atrw_anno_reid_train', 'reid_list_train.csv')
    test_annotations = osp.join(
        root, 'eval_script', 'ATRWEvalScript-main', 'annotations', 'gt_test_plain.json'
    )
    for path in [train_source, test_source, train_annotations, test_annotations]:
        if not osp.exists(path):
            raise RuntimeError('Required ATRW path not found: {}'.format(path))

    train = read_train_annotations(train_annotations)
    test_by_pid = defaultdict(list)
    for item in read_test_annotations(test_annotations):
        test_by_pid[item['pid']].append(item)

    rng = random.Random(args.seed)
    query, gallery = [], []
    for pid in sorted(test_by_pid):
        items = sorted(test_by_pid[pid], key=lambda item: item['image'])
        images = [item['image'] for item in items]
        if len(images) <= args.query_per_id:
            raise RuntimeError(
                'Identity {} has {} images; more than {} are required'.format(
                    pid, len(images), args.query_per_id
                )
            )
        selected = set(rng.sample(images, args.query_per_id))
        query_type = items[0]['query_type']
        if any(item['query_type'] != query_type for item in items):
            raise RuntimeError('Inconsistent query type for identity {}'.format(pid))

        if query_type == 'multi':
            videos = sorted({item['video'] for item in items})
            # ATRW labels some multi identities with more than two source
            # clips. Keep the requested two-camera protocol: the first clip
            # is camera 1 and the second/subsequent clips are camera 2.
            video_to_camid = {
                video: min(index + 1, 2) for index, video in enumerate(videos)
            }

        for item in items:
            if query_type == 'sing':
                camid = 0 if item['image'] in selected else 1
            else:
                camid = video_to_camid[item['video']]
            record = (pid, item['image'], camid, query_type)
            (query if item['image'] in selected else gallery).append(record)

    write_split(root, 'train', train, train_source, args.overwrite)
    write_split(root, 'query', query, test_source, args.overwrite, test_split=True)
    write_split(root, 'gallery', gallery, test_source, args.overwrite, test_split=True)
    print('ATRW prepared at {}'.format(root))
    print('train: {} images, {} IDs'.format(len(train), len({x[0] for x in train})))
    print('query: {} images, {} IDs'.format(len(query), len({x[0] for x in query})))
    print('gallery: {} images, {} IDs'.format(len(gallery), len({x[0] for x in gallery})))


if __name__ == '__main__':
    main()
