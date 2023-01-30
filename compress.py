"""Module with functions and classes for image compression"""

from __future__ import annotations
import os
import shutil

from pathlib import Path
from operator import add
from functools import reduce
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as img
import numpy as np

from PIL import Image

from util import time_func


def quad_split(image: np.ndarray):
    half = np.array_split(image, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half)
    return reduce(add, res)


def quad_concat(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray):
    top = np.concatenate((tl, tr), axis=1)
    bottom = np.concatenate((bl, br), axis=1)
    return np.concatenate((top, bottom), axis=0)


def calc_mean_color(image: np.ndarray):
    return np.mean(image, axis=(0, 1)).astype(int)


def is_one_color(image: np.ndarray):
    return all((pixel == image[0]).all() for pixel in image)


class Rectangle:
    def __init__(self, x: int, y: int, width: int, height: int):
        self._left = x
        self._top = y
        self._width = width
        self._height = height

    def get_left(self) -> int:
        return self._left

    def get_top(self) -> int:
        return self._top

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height

    def get_right(self) -> int:
        return self._left + self._width

    def get_bottom(self) -> int:
        return self._top + self._height

    def intersects(self, rect: Rectangle) -> bool:
        return not (self._left > rect.get_right() or
                    self.get_right() < rect.get_left() or
                    self._top > rect.get_bottom() or
                    self.get_bottom() < rect.get_top())


class ImageQuadTree:
    def __init__(
        self,
        image: np.ndarray,
        level: int = 0,
        rect: Rectangle = None,
        root: ImageQuadTree = None
    ):
        if rect is None:
            rect = Rectangle(0, 0, image.shape[1], image.shape[0])

        self.rect = rect
        self.level = level

        self.mean_color = calc_mean_color(image)

        self.resolution = (image.shape[1], image.shape[0])
        self.leaf = True

        if root is None:
            self.leaf_level = -1
            root = self

        self.root = root

        if root.leaf_level < level:
            root.leaf_level = level

        if not is_one_color(image):
            split_img = quad_split(image)
            self.leaf = False

            left = rect.get_left()
            top = rect.get_top()

            branches_args = [
                [
                    split_img[0],
                    level + 1,
                    Rectangle(left,
                              top,
                              split_img[0].shape[1],
                              split_img[0].shape[0]),
                    root
                ],
                [
                    split_img[1],
                    level + 1,
                    Rectangle(left + split_img[0].shape[1],
                              top,
                              split_img[1].shape[1],
                              split_img[1].shape[0]),
                    root
                ],
                [
                    split_img[2],
                    level + 1,
                    Rectangle(left,
                              top + split_img[0].shape[0],
                              split_img[2].shape[1],
                              split_img[2].shape[0]),
                    root
                ],
                [
                    split_img[3],
                    level + 1,
                    Rectangle(left + split_img[2].shape[1],
                              top + split_img[1].shape[0],
                              split_img[3].shape[1],
                              split_img[3].shape[0]),
                    root
                ]
            ]

            self.top_left = ImageQuadTree(*branches_args[0])
            self.top_right = ImageQuadTree(*branches_args[1])
            self.bottom_left = ImageQuadTree(*branches_args[2])
            self.bottom_right = ImageQuadTree(*branches_args[3])

    def get_square(self, level: int, boundaries: bool = False):
        if self.leaf or self.level == level:
            height = self.rect.get_height()
            width = self.rect.get_width()

            square = np.tile(self.mean_color, (
                height,
                width,
                1
            ))

            if boundaries:
                square[0:height, width - 1] = [0, 0, 0]
                square[height - 1, 0:width] = [0, 0, 0]

            return square

        return quad_concat(
            self.top_left.get_square(level, boundaries),
            self.top_right.get_square(level, boundaries),
            self.bottom_left.get_square(level, boundaries),
            self.bottom_right.get_square(level, boundaries)
        )

    def save_image(
        self,
        level: int,
        path: str | Path = '.',
        name: str = 'compressed',
        boundaries: bool = False,
    ):
        level = self.check_level(level)

        fig = plt.figure()
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        fig.set_size_inches((self.rect.get_width() / 100,
                             self.rect.get_height() / 100))
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(self.get_square(level, boundaries), aspect='equal')

        plt.savefig(Path(path, f'{name}.png'))

    def save_animation(
        self,
        level: int,
        path: str | Path = '.',
        name: str = 'compressed',
        boundaries: bool = False,
    ):
        level = self.check_level(level)

        frames_path = Path(os.path.dirname(__file__), '__tmp__')
        frames_path.mkdir(parents=True, exist_ok=True)

        level_range = range(self.leaf_level, level - 1, -1)
        frames = []

        for i in level_range:
            self.save_image(i, frames_path, i, boundaries)
            frames.append(Image.open(Path(frames_path, f'{i}.png')))

        frames[0].save(
            Path(path, f'{name}.gif'),
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=1000,
            loop=0
        )

        shutil.rmtree(frames_path)

    def check_level(self, level: int):
        if (level < 0):
            return 0
        if (level > self.root.leaf_level):
            return self.root.leaf_level
        return level


def compress(
    filename: str | Path,
    path: str | Path = '.',
    name: str = 'compressed',
    ratio: int = 0,
    format: Literal['img', 'anim'] = 'img',
    boundaries: bool = False,
):
    image = img.imread(filename)

    qtree = time_func(ImageQuadTree, False)(image)

    if (ratio < 0 or ratio > qtree.leaf_level):
        ratio = qtree.leaf_level

    if (format == 'img'):
        time_func(qtree.save_image, False)(ratio, path, name, boundaries)
    else:
        time_func(qtree.save_animation, False)(ratio, path, name, boundaries)
