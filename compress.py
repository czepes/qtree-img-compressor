"""Module of functions and classes for image compression"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import add
from pathlib import Path
from typing import Literal

import matplotlib.image as img
import numpy as np
from PIL import Image

from util import time_func


def compress(
    file: str | Path,
    ratio: int = 0,
    boundaries: bool = False,
    pathname: str | Path = './compressed',
    save_as: Literal['img', 'anim'] = 'img',
):
    """
    Compress image using QuadTreeImage.

    Args:
        file (str | Path): Path to file to compress.
        ratio (int, optional): Compression ratio. Defaults to 0.
        boundaries (bool, optional): Flag to add boundaries. Defaults to False.
        pathname (str | Path, optional):
            Path + name to save compressed file. Defaults to './compressed'.
        save_as (Literal['img', 'anim'], optional): Save format. Defaults to 'img'.
    """
    image = img.imread(file)

    qtree: ImageQuadTree = time_func(ImageQuadTree)(image, ratio)

    save = qtree.save_image if save_as == 'img' else qtree.save_animation

    time_func(save)(ratio, boundaries, pathname)


def quad_split(array: np.ndarray) -> np.ndarray:
    """
    Split 2D array into 4 2D arrays.

    Args:
        array (np.ndarray): 2D array.

    Returns:
        np.ndarray: Array of 4 splitted 2D arrays.
    """
    half = np.array_split(array, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half)
    return reduce(add, res)


def quad_concat(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray
) -> np.ndarray:
    """
    Concatenate 4 2D arrays into 2D array.

    Args:
        tl (np.ndarray): Top left 2D array.
        tr (np.ndarray): Top right 2D array.
        bl (np.ndarray): Bottom left 2D array.
        br (np.ndarray): Bottom right 2D array.

    Returns:
        np.ndarray: Concatenated 2D array.
    """
    top = np.concatenate((top_left, top_right), axis=1)
    bottom = np.concatenate((bottom_left, bottom_right), axis=1)
    return np.concatenate((top, bottom), axis=0)


def calc_mean_color(image: np.ndarray) -> np.ndarray[np.uint8]:
    """
    Calculate mean color from 2D array of the image.

    Args:
        image (np.ndarray): 2D array of the image.

    Returns:
        np.ndarray[np.uint8]: Mean color.
    """
    return np.mean(image, axis=(0, 1)).astype(np.uint8)


def is_mono_color(image: np.ndarray) -> bool:
    """
    Check if image consists of one color.

    Args:
        image (np.ndarray): 2D array of the image.

    Returns:
        bool: One color or not.
    """
    return all((pixel == image[0]).all() for pixel in image)


class ImageQuadTree:
    """
    Quad Tree representation of Image.
    """

    MIN_LEVEL = 0
    MAX_LEVEL = 10

    def __init__(
        self,
        image: np.ndarray,
        leaf_level: int = MAX_LEVEL,
        level: int = 0,
        rect: Rectangle = None,
    ):
        """
        Initialize Quad Tree of the Image.
        Called recursively until leaf level is reached or
        the Image of the branch consists of one color.

        Args:
            image (np.ndarray): Image array.
            leaf_level (int, optional):
                Level at which recursive construction of Quad Tree will be stopped. Defaults to 10.
            level (int, optional): Level of Quad Tree branch. Defaults to 0.
            rect (Rectangle, optional): Geometrical properties of Image. Defaults to None.
            root (ImageQuadTree, optional): Root branch of Quad Tree. Defaults to None.
        """
        if rect is None:
            rect = Rectangle(0, 0, image.shape[1], image.shape[0])

        self.rect = rect
        self.level = level

        self.mean_color = calc_mean_color(image)

        self.leaf = True

        self.leaf_level = leaf_level

        if level < leaf_level and not is_mono_color(image):
            split_img = quad_split(image)
            self.leaf = False

            left = rect.get_left()
            top = rect.get_top()

            branches_args = [
                [
                    split_img[0],
                    leaf_level,
                    level + 1,
                    Rectangle(
                        left,
                        top,
                        split_img[0].shape[1],
                        split_img[0].shape[0]
                    ),
                ],
                [
                    split_img[1],
                    leaf_level,
                    level + 1,
                    Rectangle(
                        left + split_img[0].shape[1],
                        top,
                        split_img[1].shape[1],
                        split_img[1].shape[0]
                    ),
                ],
                [
                    split_img[2],
                    leaf_level,
                    level + 1,
                    Rectangle(
                        left,
                        top + split_img[0].shape[0],
                        split_img[2].shape[1],
                        split_img[2].shape[0]
                    ),
                ],
                [
                    split_img[3],
                    leaf_level,
                    level + 1,
                    Rectangle(
                        left + split_img[2].shape[1],
                        top + split_img[1].shape[0],
                        split_img[3].shape[1],
                        split_img[3].shape[0]
                    ),
                ]
            ]

            if level == 0:
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(ImageQuadTree, *args)
                        for args in branches_args
                    ]
                    self.branches = map(
                        lambda future: future.result(), futures
                    )
            else:
                self.branches = map(
                    lambda args: ImageQuadTree(*args), branches_args
                )

    def get_square(self, level: int, boundaries: bool = False) -> np.ndarray:
        """
        Get 2D array of mean color of Quad Tree branch.

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
                square[:height, width - 1] = [0, 0, 0]
                square[height - 1, :width] = [0, 0, 0]

            return square

        return quad_concat(*map(
            lambda branch: branch.get_square(level, boundaries),
            self.branches
        ))

    def save_image(
        self,
        level: int,
        boundaries: bool = False,
        pathname: str | Path = 'compressed',
    ):
        """
        Save Quad Tree as png file.

        Args:
            level (int): Level of Quad Tree branch.
            boundaries (bool, optional): Flag to add boundaries. Defaults to False.
            pathname (str | Path, optional): Path + name to save file. Defaults to 'compressed'.
        """
        level = self.correct_level(level)

        Image \
            .fromarray(self.get_square(level, boundaries)) \
            .save(f'{pathname}.png')

    def save_animation(
        self,
        level: int,
        boundaries: bool = False,
        pathname: str | Path = 'compressed',
    ):
        """
        Save Quad Tree as gif file.

        Args:
            level (int): Level of Quad Tree branch.
            boundaries (bool, optional): Flag to add boundaries. Defaults to False.
            pathname (str | Path, optional): Path + name to save file. Defaults to 'compressed'.
        """
        level = self.correct_level(level)

        frames = []

        for i in range(level):
            frames.append(Image.fromarray(self.get_square(i, boundaries)))

        frames[0].save(
            Path(f'{pathname}.gif'),
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=1000,
            loop=0
        )

    def correct_level(self, level: int) -> int:
        """
        Check and correct level of Quad Tree branch.

        Args:
            level (int): Level of Quad Tree branch.

        Returns:
            int: Corrected level.
        """
        return min(max(level, self.MIN_LEVEL), self.leaf_level)


class Rectangle:
    """
    Rectangular geometric figure.
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Init Rectangle

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
        """
        Get left edge coordinate on the x-axis.

        Returns:
            int: Left edge coordinate on the x-axis.
        """
        return self._left

    def get_top(self) -> int:
        """
        Get top edge coordinate on the y-axis.

        Returns:
            int: Top edge coordinate on the y-axis.
        """
        return self._top

    def get_width(self) -> int:
        """
        Get Rectangle width.

        Returns:
            int: Rectangle width.
        """
        return self._width

    def get_height(self) -> int:
        """
        Get Rectangle height.

        Returns:
            int: Rectangle height.
        """
        return self._height

    def get_right(self) -> int:
        """
        Get right edge coordinate on the x-axis.

        Returns:
            int: Right edge coordinate on the x-axis.
        """
        return self._left + self._width

    def get_bottom(self) -> int:
        """
        Get bottom edge coordinate on the y-axis.

        Returns:
            int: Bottom edge coordinate on the y-axis.
        """
        return self._top + self._height

    def intersects(self, rect: Rectangle) -> bool:
        """
        Check if Rectangle intersects with other Rectangle.

        Args:
            rect (Rectangle): Other Rectangle.

        Returns:
            bool: Intersects or not.
        """
        return not (self._left > rect.get_right() or
                    self.get_right() < rect.get_left() or
                    self._top > rect.get_bottom() or
                    self.get_bottom() < rect.get_top())
