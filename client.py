"""Module for Command Line Interface"""

import argparse
import os.path

from compress import compress


def main():
    parser = argparse.ArgumentParser(
        prog='QTreeImageCompressor',
        description='Program compresses images with QuadTree recursive abstraction',
    )

    parser.add_argument('file', help='path to the image to be compressed')
    parser.add_argument(
        '-p', '--path',
        default='.',
        help='path to the directory to save compression result [default: current dir]',
    )
    parser.add_argument(
        '-n', '--name',
        help='name of the file to save compression result [default: [file]-comp]',
    )
    parser.add_argument(
        '-r', '--ratio',
        type=int, default=0,
        help='the higher the value, the lower the compression ratio [default: 0]'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['img', 'anim'],
        default='img',
        help='format in which to save result file [default: img]'
    )
    parser.add_argument(
        '-b', '--bounds',
        dest='boundaries',
        action='store_true',
        help='show QuadTree boundaries'
    )

    args = parser.parse_args()

    name = args.name
    if (not name):
        name = os.path.splitext(
            os.path.basename(args.file)
        )[0] + '-comp'

    print('Compression started!')
    compress(
        args.file,
        args.path,
        name,
        args.ratio,
        args.format,
        args.boundaries,
    )
    print('Compression finished!')


if __name__ == '__main__':
    main()
