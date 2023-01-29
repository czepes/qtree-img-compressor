"""Module for Command Line Interface"""

import argparse

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
        help='path to dir for saving result image (default = current dir)',
    )
    parser.add_argument(
        '-n', '--name',
        default='compressed',
        help='name of the file for saving (default = compressed)',
    )
    parser.add_argument(
        '-b', '--bounds',
        dest='boundaries',
        action='store_true',
        help='show QuadTree boundaries'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['png', 'gif'],
        default='png',
        help='format in which to save result file (default = png)'
    )
    parser.add_argument(
        '-r', '--ratio',
        type=int, default=-1,
        help='the higher the value, the lower the compression ratio (default = min compression)'
    )

    args = parser.parse_args()

    print('Compression started!')
    compress(args.filename, args.path, args.name, args.boundaries, args.format, args.ratio)
    print('Compression finished!')


def compress(filename, path, name, boundaries, format, ratio):
    image = img.imread(filename)

    qtree = time_func(ImageQuadTree, False)(image)
    
    if (ratio < 0 or ratio > qtree.leaf_level):
        ratio = qtree.leaf_level

    if (format == 'png'):
        qtree.save_image(ratio, boundaries, name, path)
    else:
        qtree.save_gif(ratio, boundaries, name, path)


if __name__ == '__main__':
    main()
