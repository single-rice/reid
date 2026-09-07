from __future__ import absolute_import, print_function

import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
VIDEO_PATTERN = re.compile(r'(video_\d+)', re.IGNORECASE)
PID_PATTERN = re.compile(r'^tiger_(\d+)$', re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare the Wild_Tiger dataset')
    parser.add_argument('--source', required=True, help='directory containing one folder per tiger')
    parser.add_argument('--destination', required=True, help='output Wild_Tiger directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-multi-ids', type=int, default=30)
    parser.add_argument('--test-single-ids', type=int, default=50)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def images_in(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def video_names(images):
    names = set()
    for image in images:
        match = VIDEO_PATTERN.search(image.stem)
        if match is None:
            raise RuntimeError(
                'Cannot parse video name from {}'.format(image)
            )
        names.add(match.group(1).lower())
    return names


def tiger_pid(tiger_id):
    match = PID_PATTERN.match(tiger_id)
    if match is None:
        raise RuntimeError(
            'Invalid identity directory {}; expected tiger_<number>'.format(
                tiger_id
            )
        )
    return int(match.group(1))


def split_identities(identities, rng, num_multi, num_single):
    multi = sorted(
        (item for item in identities if len(video_names(item[1])) > 1),
        key=lambda item: item[0]
    )
    single = sorted(
        (item for item in identities if len(video_names(item[1])) == 1),
        key=lambda item: item[0]
    )
    if len(multi) < num_multi:
        raise RuntimeError(
            'Requested {} multi-video test identities, but only {} are '
            'available'.format(num_multi, len(multi))
        )
    if len(single) < num_single:
        raise RuntimeError(
            'Requested {} single-video test identities, but only {} are '
            'available'.format(num_single, len(single))
        )

    rng.shuffle(multi)
    rng.shuffle(single)
    test = multi[:num_multi] + single[:num_single]
    train_val = multi[num_multi:] + single[num_single:]
    return (
        sorted(train_val, key=lambda item: item[0]),
        sorted(test, key=lambda item: item[0])
    )


def split_train_val(images, rng):
    shuffled = list(images)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(0.2 * len(shuffled))))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def copy_images(images, destination, tiger_id):
    destination.mkdir(parents=True, exist_ok=True)
    used = set()
    copied_names = []
    for index, source in enumerate(images):
        name = source.name
        if name in used or (destination / name).exists():
            name = '{}_{:05d}{}'.format(tiger_id, index, source.suffix.lower())
        used.add(name)
        shutil.copy2(str(source), str(destination / name))
        copied_names.append('{}/{}'.format(tiger_id, name))
    return copied_names


def build_test_records(pid, query, gallery):
    all_images = list(query) + list(gallery)
    videos = sorted(video_names(all_images))
    query_type = 'sing' if len(videos) == 1 else 'multi'
    video2camid = {
        video: index + 1 for index, video in enumerate(videos)
    }

    def _records(images, is_query):
        records = []
        for image in images:
            if query_type == 'sing':
                camid = 0 if is_query else 1
            else:
                match = VIDEO_PATTERN.search(image.stem)
                camid = video2camid[match.group(1).lower()]
            records.append((pid, image, camid, query_type))
        return records

    return _records(query, True), _records(gallery, False), video2camid


def write_csv(path, rows, test_split=False):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        if test_split:
            writer.writerow(['pid', 'image', 'camid', 'query'])
        else:
            writer.writerow(['pid', 'image'])
        writer.writerows(rows)


def split_query_gallery(images, rng):
    images = list(images)
    if len(images) < 3:
        raise RuntimeError(
            'Each test identity needs at least 3 images: '
            '2 query and 1 gallery'
        )

    rng.shuffle(images)
    query = images[:2]
    gallery = images[2:]

    video_counts = {}
    for image in images:
        match = VIDEO_PATTERN.search(image.stem)
        video = match.group(1).lower() if match else 'unknown'
        video_counts[video] = video_counts.get(video, 0) + 1
    return query, gallery, video_counts


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    if not source.is_dir():
        raise RuntimeError('Source directory does not exist: {}'.format(source))
    if destination.exists():
        if not args.overwrite:
            raise RuntimeError('Destination exists; pass --overwrite to replace it: {}'.format(destination))
        shutil.rmtree(str(destination))

    identities = [(path.name, images_in(path)) for path in source.iterdir() if path.is_dir()]
    identities = [(name, images) for name, images in identities if images]
    if len(identities) < 3:
        raise RuntimeError('At least three non-empty identity folders are required')

    split_rng = random.Random(args.seed)
    image_rng = random.Random(args.seed + 1)
    train_val_identities, test_identities = split_identities(
        identities,
        split_rng,
        num_multi=args.test_multi_ids,
        num_single=args.test_single_ids
    )

    manifest = {
        'seed': args.seed,
        'test_identity_selection': {
            'multi_video': args.test_multi_ids,
            'single_video': args.test_single_ids
        },
        'ratios': {'train_val_ids': 0.8, 'test_ids': 0.2,
                   'train_images': 0.8, 'val_images': 0.2},
        'splits': {'train': [], 'val': [], 'test': []}
    }
    train_rows = []
    val_rows = []
    query_rows = []
    gallery_rows = []
    for tiger_id, images in train_val_identities:
        train_images, val_images = split_train_val(images, image_rng)
        train_names = copy_images(
            train_images, destination / 'train' / tiger_id, tiger_id
        )
        val_names = copy_images(
            val_images, destination / 'val' / tiger_id, tiger_id
        )
        pid = tiger_pid(tiger_id)
        train_rows.extend((pid, name) for name in train_names)
        val_rows.extend((pid, name) for name in val_names)
        manifest['splits']['train'].append(
            {'id': tiger_id, 'images': len(train_images)}
        )
        manifest['splits']['val'].append(
            {'id': tiger_id, 'images': len(val_images)}
        )

    for tiger_id, images in test_identities:
        query, gallery, video_counts = split_query_gallery(images, image_rng)
        query_names = copy_images(
            query, destination / 'test' / 'query' / tiger_id, tiger_id
        )
        gallery_names = copy_images(
            gallery, destination / 'test' / 'gallery' / tiger_id, tiger_id
        )
        pid = tiger_pid(tiger_id)
        query_records, gallery_records, video2camid = build_test_records(
            pid, query, gallery
        )
        query_rows.extend(
            (record[0], name, record[2], record[3])
            for record, name in zip(query_records, query_names)
        )
        gallery_rows.extend(
            (record[0], name, record[2], record[3])
            for record, name in zip(gallery_records, gallery_names)
        )
        manifest['splits']['test'].append(
            {
                'id': tiger_id,
                'videos': video_counts,
                'video2camid': video2camid,
                'query_type': query_records[0][3],
                'query_images': len(query),
                'gallery_images': len(gallery)
            }
        )

    write_csv(destination / 'train.csv', train_rows)
    write_csv(destination / 'val.csv', val_rows)
    write_csv(destination / 'query.csv', query_rows, test_split=True)
    write_csv(destination / 'gallery.csv', gallery_rows, test_split=True)
    with (destination / 'split.json').open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print('Prepared {} identities: train/val={}, test={}'.format(
        len(identities), len(train_val_identities), len(test_identities)
    ))
    print('Test identities: multi-video={}, single-video={}'.format(
        args.test_multi_ids, args.test_single_ids
    ))


if __name__ == '__main__':
    main()
