"""Module of functions and classes for image compression"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import add
import os
from pathlib import Path
import shutil
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from PIL import Image

from util import time_func


def compress(
    file: str | Path,
    path: str | Path = '.',
    name: str = 'compressed',
    ratio: int = 0,
    format: Literal['img', 'anim'] = 'img',
    boundaries: bool = False,
):
    """Compress image using QuadTreeImage.

    Args:
        filename (str | Path): Path to file to compress.
        path (str | Path, optional): Path to save compressed file. Defaults to '.'.
        name (str, optional): File name to save. Defaults to 'compressed'.
        ratio (int, optional): Compression ratio. Defaults to 0.
        format (Literal['img', 'anim'], optional): Compression format. Defaults to 'img'.
        boundaries (bool, optional): Flag to add boundaries. Defaults to False.
    """
    image = img.imread(file)

    qtree = time_func(ImageQuadTree, False)(image)

    if (ratio < 0 or ratio > qtree.leaf_level):
        ratio = qtree.leaf_level

    if (format == 'img'):
        time_func(qtree.save_image, False)(ratio, path, name, boundaries)
    else:
        time_func(qtree.save_animation, False)(ratio, path, name, boundaries)


def quad_split(array: np.ndarray) -> list:
    """Split 2D array into 4 2D arrays.

    Args:
        array (np.ndarray): 2D array.

    Returns:
        list: List of 4 splitted 2D arrays.
    """    
    half = np.array_split(array, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half)
    return reduce(add, res)


def quad_concat(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray) -> np.ndarray:
    """Concatenate 4 2D arrays into 2D array.

    Args:
        tl (np.ndarray): Top left 2D array.
        tr (np.ndarray): Top right 2D array.
        bl (np.ndarray): Bottom left 2D array.
        br (np.ndarray): Bottom right 2D array.

    Returns:
        np.ndarray: Concatenated 2D array.
    """    
    top = np.concatenate((tl, tr), axis=1)
    bottom = np.concatenate((bl, br), axis=1)
    return np.concatenate((top, bottom), axis=0)


def calc_mean_color(image: np.ndarray) -> np.ndarray[int, int, int]:
    """Calculate mean color from 2D array of the image. 

    Args:
        image (np.ndarray): 2D array of the image.

    Returns:
        np.ndarray[int, int, int]: Mean color.
    """
    return np.mean(image, axis=(0, 1)).astype(int)


def is_mono_color(image: np.ndarray) -> bool:
    """Check if image consists of one color.

    Args:
        image (np.ndarray): 2D array of the image.

    Returns:
        bool: One color or not.
    """    
    return all((pixel == image[0]).all() for pixel in image)


class ImageQuadTree:
    def __init__(
        self,
        image: np.ndarray,
        level: int = 0,
        rect: Rectangle = None,
        root: ImageQuadTree = None
    ):
        """
        Initialize Quad Tree of the Image.
        Called recursively until the Image of the branch consists of one color.

        Args:
            image (np.ndarray): Image array.
            level (int, optional): Level of Quad Tree branch. Defaults to 0.
            rect (Rectangle, optional): Geometrical properties of Image. Defaults to None.
            root (ImageQuadTree, optional): Root branch of Quad Tree. Defaults to None.
        """
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

        if not is_mono_color(image):
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

            if level == 0:
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(ImageQuadTree, *args)
                               for args in branches_args]
                    self.top_left = futures[0].result()
                    self.top_right = futures[1].result()
                    self.bottom_left = futures[2].result()
                    self.bottom_right = futures[3].result()
            else:
                self.top_left = ImageQuadTree(*branches_args[0])
                self.top_right = ImageQuadTree(*branches_args[1])
                self.bottom_left = ImageQuadTree(*branches_args[2])
                self.bottom_right = ImageQuadTree(*branches_args[3])

    def get_square(self, level: int, boundaries: bool = False) -> np.ndarray:
        """Get 2D array of mean color of Quad Tree branch.

        Args:
            level (int): Level of Quad Tree branch.
            boundaries (bool, optional): Flag to add boundaries. Defaults to False.

        Returns:
            np.ndarray: 2D array of colors.
        """
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
        """Save Quad Tree as png file.

        Args:
            level (int): Level of Quad Tree branch.
            path (str | Path, optional): Path to save file. Defaults to '.'.
            name (str, optional): File name to save. Defaults to 'compressed'.
            boundaries (bool, optional): Flag to add boundaries. Defaults to False.
        """
        level = self.correct_level(level)

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
        """Save Quad Tree as gif file.

        Args:
            level (int): Level of Quad Tree branch.
            path (str | Path, optional): Path to save file. Defaults to '.'. Defaults to '.'.
            name (str, optional): File name to save. Defaults to 'compressed'.
            boundaries (bool, optional): Flag to add boundaries. Defaults to False.
        """
        level = self.correct_level(level)

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

    def correct_level(self, level: int) -> int:
        """Check and correct level of Quad Tree branch .

        Args:
            level (int): Level of Quad Tree branch.

        Returns:
            int: Corrected level.
        """
        if (level < 0):
            return 0
        if (level > self.root.leaf_level):
            return self.root.leaf_level
        return level


class Rectangle:
    def __init__(self, x: int, y: int, width: int, height: int):
        """Init Rectangle

        Args:
            x (int): Top left corner coordinate on the x-axis.
            y (int): Top left corner coordinate on the y-axis.
            width (int): Rectangle width
            height (int): Reactange height
        """
        self._left = x
        self._top = y
        self._width = width
        self._height = height

    def get_left(self) -> int:
        """Get left edge coordinate on the x-axis. 

        Returns:
            int: Left edge coordinate on the x-axis.
        """
        return self._left

    def get_top(self) -> int:
        """Get top edge coordinate on the y-axis. 

        Returns:
            int: Top edge coordinate on the y-axis.
        """
        return self._top

    def get_width(self) -> int:
        """Get Rectangle width.

        Returns:
            int: Rectangle width.
        """
        return self._width

    def get_height(self) -> int:
        """Get Rectangle height.

        Returns:
            int: Rectangle height.
        """
        return self._height

    def get_right(self) -> int:
        """Get right edge coordinate on the x-axis. 

        Returns:
            int: Right edge coordinate on the x-axis.
        """
        return self._left + self._width

    def get_bottom(self) -> int:
        """Get bottom edge coordinate on the y-axis. 

        Returns:
            int: Bottom edge coordinate on the y-axis.
        """
        return self._top + self._height

    def intersects(self, rect: Rectangle) -> bool:
        """Check if Rectangle intersects with other Rectangle.

        Args:
            rect (Rectangle): Other Rectangle.

        Returns:
            bool: Intersects or not.
        """
        return not (self._left > rect.get_right() or
                    self.get_right() < rect.get_left() or
                    self._top > rect.get_bottom() or
                    self.get_bottom() < rect.get_top())
