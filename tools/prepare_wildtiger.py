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


def split_counts(n):
    # Largest-remainder allocation gives the closest integer 60/20/20 split.
    raw = [0.6 * n, 0.2 * n, 0.2 * n]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[:n - sum(counts)]:
        counts[index] += 1
    return counts


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
    n_train, n_val, _ = split_counts(len(identities))
    partitions = {
        'train': identities[:n_train],
        'val': identities[n_train:n_train + n_val],
        'test': identities[n_train + n_val:],
    }

    manifest = {'seed': args.seed, 'ratios': {'train': 0.6, 'val': 0.2, 'test': 0.2}, 'splits': {}}
    for split in ('train', 'val'):
        manifest['splits'][split] = []
        for tiger_id, images in partitions[split]:
            copy_images(images, destination / split / tiger_id, tiger_id)
            manifest['splits'][split].append({'id': tiger_id, 'images': len(images)})

    manifest['splits']['test'] = []
    for tiger_id, images in partitions['test']:
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
    print('Prepared {} identities: train={}, val={}, test={}'.format(
        len(identities), len(partitions['train']), len(partitions['val']), len(partitions['test'])
    ))


if __name__ == '__main__':
    main()
