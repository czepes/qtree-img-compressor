"""Module with functions and classes for image compression"""

from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as img
import numpy as np

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image

from pathlib import Path

from operator import add
from functools import reduce

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
    return np.mean(image, axis=(0, 1))


def calc_mean_color_split(split_image: np.ndarray):
    return np.array(list(map(
        lambda x: calc_mean_color(x), split_image)
    )).astype(int)


def is_equal(my_list):
    return all((x == my_list[0]).all() for x in my_list)


def color_is_equal(image, colors):
    return all((x == colors[0]).all() for x in calc_mean_color_split(image))


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
        color_type: type = None,
        root: ImageQuadTree = None
    ):
        if rect is None:
            rect = Rectangle(0, 0, image.shape[1], image.shape[0])

        self.rect = rect
        self.level = level

        if not color_type:
            color_type = type(image[0][0][0].item())
        self.mean_color = calc_mean_color(image).astype(color_type)

        self.resolution = (image.shape[1], image.shape[0])
        self.leaf = True

        if root is None:
            self.leaf_level = -1
            root = self
        if root.leaf_level < level:
            root.leaf_level = level

        if not is_equal(image):
            split_img = quad_split(image)
            self.leaf = False

            left = rect.get_left()
            top = rect.get_top()

            arguments = [
                [
                    split_img[0],
                    level + 1,
                    Rectangle(left,
                              top,
                              split_img[0].shape[1],
                              split_img[0].shape[0]),
                    color_type,
                    root
                ],
                [
                    split_img[1],
                    level + 1,
                    Rectangle(left + split_img[0].shape[1],
                              top,
                              split_img[1].shape[1],
                              split_img[1].shape[0]),
                    color_type,
                    root
                ],
                [
                    split_img[2],
                    level + 1,
                    Rectangle(left,
                              top + split_img[0].shape[0],
                              split_img[2].shape[1],
                              split_img[2].shape[0]),
                    color_type,
                    root
                ],
                [
                    split_img[3],
                    level + 1,
                    Rectangle(left + split_img[2].shape[1],
                              top + split_img[1].shape[0],
                              split_img[3].shape[1],
                              split_img[3].shape[0]),
                    color_type,
                    root
                ]
            ]

            # if level == 0:
            #     with ProcessPoolExecutor() as executor:
            #         futures = [executor.submit(ImageQuadTree, *args)
            #                    for args in arguments]
            #         self.top_left = futures[0].result()
            #         self.top_right = futures[1].result()
            #         self.bottom_left = futures[2].result()
            #         self.bottom_right = futures[3].result()
            # else:
            self.top_left = ImageQuadTree(*arguments[0])
            self.top_right = ImageQuadTree(*arguments[1])
            self.bottom_left = ImageQuadTree(*arguments[2])
            self.bottom_right = ImageQuadTree(*arguments[3])

    def get_square(self, level: int, boundaries: bool = False):
        if self.leaf or self.level == level:
            height = self.rect.get_height()
            width = self.rect.get_width()

            square = np.tile(self.mean_color, (
            # ? Width -> height or height -> width?
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

    @time_func
    def save_image(
        self,
        level: int,
        path: str = '.',
        name: str = 'compressed',
        boundaries: bool = False,
    ):
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

        plt.savefig(Path(f'{path}/{name}.png'))

    @time_func
    def save_animation(
        self,
        level: int,
        path: str = '.',
        name: str = 'compressed',
        boundaries: bool = False,
    ):
        Path(f'tmp').mkdir(parents=True, exist_ok=True)

        img_range = range(self.leaf_level, level - 1, -1)

        for i in img_range:
            self.save_image(i, 'tmp', i, boundaries)

        frames = []

        for i in img_range:
            frames.append(Image.open(f'tmp/{i}.png'))

        frames[0].save(
            Path(f'{path}/{name}.gif'),
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=1000,
            loop=0
        )

        shutil.rmtree('tmp')
