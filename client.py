"""Module for Command Line Interface"""

import argparse
from os.path import splitext

import matplotlib.image as img

from compress import ImageQuadTree
from util import time_func


def main():
    parser = argparse.ArgumentParser(
        prog='QTreeImageCompressor',
        description='Program compresses images with QuadTree recursive abstraction',
    )

    parser.add_argument('filename', help='image to compress')
    parser.add_argument(
        '-p', '--path',
        default='.',
        help='path to dir to save compression result [default: current dir]',
    )
    parser.add_argument(
        '-n', '--name',
        help='name of the file to save compression result [default: [filename]-comp]',
    )
    parser.add_argument(
        '-b', '--bounds',
        dest='boundaries',
        action='store_true',
        help='show QuadTree boundaries'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['img', 'anim'],
        default='img',
        help='format in which to save result file [default: img]'
    )
    parser.add_argument(
        '-r', '--ratio',
        type=int, default=0,
        help='the higher the value, the lower the compression ratio [default: 0]'
    )

    args = parser.parse_args()

    name = args.name
    if (not name):
        name = splitext(args.filename)[0] + '-comp'

    print('Compression started!')
    compress(
        args.filename,
        args.path,
        name,
        args.ratio,
        args.format,
        args.boundaries,
    )
    print('Compression finished!')


def compress(
    filename: str,
    path: str,
    name: str,
    ratio: int,
    format: str,
    boundaries: bool,
):
    image = img.imread(filename)

    qtree = time_func(ImageQuadTree, False)(image)

    if (ratio < 0 or ratio > qtree.leaf_level):
        ratio = qtree.leaf_level

    if (format == 'img'):
        qtree.save_image(ratio, path, name, boundaries)
    else:
        qtree.save_animation(ratio, path, name, boundaries)


if __name__ == '__main__':
    main()
