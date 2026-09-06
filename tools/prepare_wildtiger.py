from __future__ import absolute_import, print_function

import argparse
import json
import random
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
VIDEO_PATTERN = re.compile(r'(video_\d+)', re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare the Wild_Tiger dataset')
    parser.add_argument('--source', required=True, help='directory containing one folder per tiger')
    parser.add_argument('--destination', required=True, help='output Wild_Tiger directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def images_in(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
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
    for index, source in enumerate(images):
        name = source.name
        if name in used or (destination / name).exists():
            name = '{}_{:05d}{}'.format(tiger_id, index, source.suffix.lower())
        used.add(name)
        shutil.copy2(str(source), str(destination / name))


def split_query_gallery(images, rng):
    by_video = {}
    for image in images:
        match = VIDEO_PATTERN.search(image.stem)
        video = match.group(1).lower() if match else 'unknown'
        by_video.setdefault(video, []).append(image)

    query, gallery = [], []
    video_counts = {}
    for video in sorted(by_video):
        video_images = list(by_video[video])
        if len(video_images) < 3:
            raise RuntimeError(
                'Video {} needs at least 3 images: 2 query and 1 gallery'.format(video)
            )
        rng.shuffle(video_images)
        query.extend(video_images[:2])
        gallery.extend(video_images[2:])
        video_counts[video] = len(video_images)
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

    rng = random.Random(args.seed)
    identities.sort(key=lambda item: item[0])
    rng.shuffle(identities)
    # 403 identities -> 323 train/validation identities and 80 held-out test
    # identities. The explicit floor keeps the requested held-out set at 80.
    n_test = int(0.2 * len(identities))
    train_val_identities = identities[:-n_test]
    test_identities = identities[-n_test:]

    manifest = {
        'seed': args.seed,
        'ratios': {'train_val_ids': 0.8, 'test_ids': 0.2,
                   'train_images': 0.8, 'val_images': 0.2},
        'splits': {'train': [], 'val': [], 'test': []}
    }
    for tiger_id, images in train_val_identities:
        train_images, val_images = split_train_val(images, rng)
        copy_images(train_images, destination / 'train' / tiger_id, tiger_id)
        copy_images(val_images, destination / 'val' / tiger_id, tiger_id)
        manifest['splits']['train'].append(
            {'id': tiger_id, 'images': len(train_images)}
        )
        manifest['splits']['val'].append(
            {'id': tiger_id, 'images': len(val_images)}
        )

    for tiger_id, images in test_identities:
        query, gallery, video_counts = split_query_gallery(images, rng)
        copy_images(query, destination / 'test' / 'query' / tiger_id, tiger_id)
        copy_images(gallery, destination / 'test' / 'gallery' / tiger_id, tiger_id)
        manifest['splits']['test'].append(
            {
                'id': tiger_id,
                'videos': video_counts,
                'query_images': len(query),
                'gallery_images': len(gallery)
            }
        )

    with (destination / 'split.json').open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print('Prepared {} identities: train/val={}, test={}'.format(
        len(identities), len(train_val_identities), len(test_identities)
    ))


if __name__ == '__main__':
    main()
