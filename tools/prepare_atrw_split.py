from __future__ import print_function, absolute_import

import argparse
import csv
import os
import os.path as osp
import shutil
from collections import defaultdict


def read_annotations(csv_path, img_dir):
    pid_to_imgs = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            pid = int(row[0])
            img_name = row[1].strip()
            src = osp.join(img_dir, img_name)
            if not osp.isfile(src):
                raise RuntimeError('Missing image listed in CSV: {}'.format(src))
            pid_to_imgs[pid].append(img_name)

    for pid in pid_to_imgs:
        pid_to_imgs[pid] = sorted(pid_to_imgs[pid])
    return pid_to_imgs


def write_rows(rows, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pid', 'image'])
        writer.writerows(rows)


def copy_split(rows, img_dir, split_dir):
    os.makedirs(split_dir, exist_ok=True)
    for pid, img_name in rows:
        src = osp.join(img_dir, img_name)
        dst = osp.join(split_dir, img_name)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='reid-data/ATRW')
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--out-name', default='split_6_4')
    args = parser.parse_args()

    root = osp.abspath(args.root)
    img_dir = osp.join(root, 'atrw_reid_train', 'train')
    csv_path = osp.join(root, 'atrw_anno_reid_train', 'reid_list_train.csv')
    out_dir = osp.join(root, args.out_name)

    pid_to_imgs = read_annotations(csv_path, img_dir)
    pids = sorted(pid_to_imgs)
    num_train_pids = int(len(pids) * args.train_ratio)
    num_train_pids = max(1, min(num_train_pids, len(pids) - 1))

    train_pids = pids[:num_train_pids]
    test_pids = pids[num_train_pids:]

    train_rows = []
    for pid in train_pids:
        train_rows.extend((pid, img_name) for img_name in pid_to_imgs[pid])

    query_rows = []
    gallery_rows = []
    for pid in test_pids:
        img_names = pid_to_imgs[pid]
        query_rows.append((pid, img_names[0]))
        gallery_rows.extend((pid, img_name) for img_name in img_names[1:])

    os.makedirs(out_dir, exist_ok=True)
    copy_split(train_rows, img_dir, osp.join(out_dir, 'train'))
    copy_split(query_rows, img_dir, osp.join(out_dir, 'query'))
    copy_split(gallery_rows, img_dir, osp.join(out_dir, 'gallery'))

    write_rows(train_rows, osp.join(out_dir, 'train.csv'))
    write_rows(query_rows, osp.join(out_dir, 'query.csv'))
    write_rows(gallery_rows, osp.join(out_dir, 'gallery.csv'))

    print('Output: {}'.format(out_dir))
    print('train:   {} ids, {} images'.format(len(train_pids), len(train_rows)))
    print('query:   {} ids, {} images'.format(len(test_pids), len(query_rows)))
    print('gallery: {} ids, {} images'.format(len(test_pids), len(gallery_rows)))


if __name__ == '__main__':
    main()
