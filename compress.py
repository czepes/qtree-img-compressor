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


def quad_split(image):
    half = np.array_split(image, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half)
    return reduce(add, res)


def quad_concat(tl, tr, bl, br):
    top = np.concatenate((tl, tr), axis=1)
    bottom = np.concatenate((bl, br), axis=1)
    return np.concatenate((top, bottom), axis=0)


def calc_mean(image):
    return np.mean(image, axis=(0, 1))


def calc_mean_color(split_image):
    return np.array(list(map(
        lambda x: calc_mean(x), split_image))).astype(int)


def is_equal(my_list):
    return all((x == my_list[0]).all() for x in my_list)


def color_is_equal(image):
    return all((x == colors[0]).all() for x in calc_mean_color(image))


class ImageQuadTree:
    def __init__(self, image, level=0, boundary=None, color_type=None, root=None):
        if boundary is None:
            boundary = Rectangle(0, 0, image.shape[1], image.shape[0])
        self.boundary = boundary
        self.level = level
        if not color_type:
            color_type = type(image[0][0][0].item())
        self.mean = calc_mean(image).astype(color_type)
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
            arguments = [
                [
                    split_img[0],
                    level + 1,
                    Rectangle(boundary.get_left(),
                              boundary.get_top(),
                              split_img[0].shape[1],
                              split_img[0].shape[0]),
                    color_type
                ],
                [
                    split_img[1],
                    level + 1,
                    Rectangle(boundary.get_left() + split_img[0].shape[1],
                              boundary.get_top(),
                              split_img[1].shape[1],
                              split_img[1].shape[0]),
                    color_type
                ],
                [
                    split_img[2],
                    level + 1,
                    Rectangle(boundary.get_left(),
                              boundary.get_top() + split_img[0].shape[0],
                              split_img[2].shape[1],
                              split_img[2].shape[0]),
                    color_type
                ],
                [
                    split_img[3],
                    level + 1,
                    Rectangle(boundary.get_left() + split_img[2].shape[1],
                              boundary.get_top() + split_img[1].shape[0],
                              split_img[3].shape[1],
                              split_img[3].shape[0]),
                    color_type
                ]
            ]
            # if level == 0:
            #     with ProcessPoolExecutor() as executor:
            #         futures = [executor.submit(ImageQuadTree, *args, root)
            #                    for args in arguments]
            #         self.top_left = futures[0].result()
            #         self.top_right = futures[1].result()
            #         self.bottom_left = futures[2].result()
            #         self.bottom_right = futures[3].result()
            # else:
            self.top_left = ImageQuadTree(*arguments[0], root)
            self.top_right = ImageQuadTree(*arguments[1], root)
            self.bottom_left = ImageQuadTree(*arguments[2], root)
            self.bottom_right = ImageQuadTree(*arguments[3], root)

    def get_image(self, level):
        if self.leaf or self.level == level:
            return np.tile(self.mean, (
                self.boundary.get_height(),
                self.boundary.get_width(),
                1
            ))
        return quad_concat(
            self.top_left.get_image(level),
            self.top_right.get_image(level),
            self.bottom_left.get_image(level),
            self.bottom_right.get_image(level)
        )

    def draw_boundaries(self, ax, level):
        if self.leaf or self.level == level:
            line_width = 0.5
            x1 = self.boundary.get_left()
            x2 = self.boundary.get_right() - line_width
            y1 = self.boundary.get_top()
            y2 = self.boundary.get_bottom() - line_width
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                    c='black', lw=line_width)
            return
        self.top_left.draw_boundaries(ax, level)
        self.top_right.draw_boundaries(ax, level)
        self.bottom_left.draw_boundaries(ax, level)
        self.bottom_right.draw_boundaries(ax, level)

    @time_func
    def save_image(
        self,
        level: int,
        boundaries: bool = False,
        name: str = 'compressed',
        path: str = '.',
    ):
        if self.leaf:
            return

        fig = plt.figure()
        plt.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        fig.set_size_inches((self.boundary.get_width() / 100,
                             self.boundary.get_height() / 100))
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(self.get_image(level), aspect='equal')
        
        if boundaries and level < 10:
            time_func(self.draw_boundaries)(ax, level)

        plt.savefig(Path(f'{path}/{name}.png'))

    @time_func
    def save_gif(
        self,
        level: int,
        boundaries: bool = False,
        name: str = 'compressed',
        path: str = '.',
    ):
        Path(f'tmp').mkdir(parents=True, exist_ok=True)

        for i in range(level + 1):
            self.save_image(i, boundaries, i, 'tmp')

        frames = []

        for i in range(level + 1):
            frames.append(Image.open(f'tmp/{i}.png'))

        frames[0].save(
            Path(f'{name}.gif'),
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=1000,
            loop=0
        )

        shutil.rmtree('tmp')


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
