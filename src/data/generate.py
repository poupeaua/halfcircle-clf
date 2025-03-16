"""
Code to generate half circle images
"""

import random
from typing import Optional

import numpy as np
import cv2
import pymupdf

from src.data.config import IMG_SHAPE, MAX_RADIUS, MIN_RADIUS, \
    MIN_THICKNESS, MAX_THICKNESS, PICSUM_URL
from src.data.utils import download_image
from src.data.readfile import read_pdf_to_images


def generate_half_circle_image(
        image: Optional[np.ndarray] = None,
        width: int = IMG_SHAPE[1], 
        height: int = IMG_SHAPE[0],
        min_radius: int = MIN_RADIUS,
        max_radius: int = MAX_RADIUS,
        min_thickness: int = MIN_THICKNESS,
        max_thickness: int = MAX_THICKNESS,
        x_center_offset_pct: int = 0.2,
        y_center_offset_pct: int = 0.2,
        min_angle: int = -15,
        max_angle: int = 15,
        added_thickness_cleaning_image: Optional[int] = None
) -> np.ndarray:
    """Generate a random half circle image with a half circle in a random position.
    You can specify how random the half circle will be drawn.

    Args:
        image (np.ndarray, optional): image to use to draw a half circle on top of. 
            Defaults to None which means we initialize a white image.
        width (int, optional): width of the image. Defaults to IMG_SHAPE[1].
        height (int, optional): height of the image. Defaults to IMG_SHAPE[0].
        min_radius (int, optional): minimum radius of the half circle. 
            Defaults to MIN_RADIUS.
        max_radius (int, optional): maximum radius of the half circle. 
            Defaults to MAX_RADIUS.
        min_thickness (int, optional): minimum thickness of the hlaf circle contour. 
            Defaults to MIN_THICKNESS.
        max_thickness (int, optional): maximum thickness of the hlaf circle contour. 
            Defaults to MAX_THICKNESS.
        x_center_offset_pct (int, optional): x shift of the center in percentage of the
            image width. Defaults to 0.2.
        y_center_offset_pct (int, optional): y shift of the center in percentage of the
            image height. Defaults to 0.2.
        min_angle (int, optional): minimum angle of the half circle. Defaults to -15.
        max_angle (int, optional): maximum angle of the half circle. Defaults to 15.

    Returns:
        np.ndarray: image with a half circle
    """
    if image is not None:
        width = image.shape[1]
        height = image.shape[0]

    # center of the half circle configuration
    width_offset = int(width * random.uniform(-x_center_offset_pct, x_center_offset_pct))
    height_offset = int(height * random.uniform(-y_center_offset_pct, y_center_offset_pct))
    center = (width // 2 + width_offset, height // 2 + height_offset)

    # radius configuration
    absolute_min_limit_radius = min(width, height) // 2
    chose_radius = random.randint(min_radius, max_radius)
    radius = min(absolute_min_limit_radius, chose_radius)

    # angle configuration
    start_angle = random.randint(min_angle, max_angle) + 180
    end_angle = start_angle + 180

    # thickness configuration
    thickness = random.randint(min_thickness, max_thickness)

    if image is not None:
        # draw a white half-circle on top of the image 
        # before adding the black half-circle
        cv2.ellipse(
            img=image, 
            center=center, 
            axes=(radius, radius), 
            angle=0, 
            startAngle=start_angle, 
            endAngle=end_angle, 
            color=255, 
            thickness=thickness+added_thickness_cleaning_image
        )
    else:
        # initialize a white image
        image = np.full(shape=(height, width), fill_value=255, dtype=np.uint8)

    # add half circle in image
    cv2.ellipse(
        img=image, 
        center=center, 
        axes=(radius, radius), 
        angle=0, 
        startAngle=start_angle, 
        endAngle=end_angle, 
        color=0, 
        thickness=thickness
    )

    return image

def generate_filled_image(
        width: int = IMG_SHAPE[1], 
        height: int = IMG_SHAPE[0],
        fill_value: int = 255
) -> np.ndarray:
    """Generate a filled image

    Args:
        width (int, optional): width of the image. Defaults to IMG_SHAPE[1].
        height (int, optional): height of the image. Defaults to IMG_SHAPE[0].
        fill_value (int, optional): fill value. Defaults to 255.

    Returns:
        np.ndarray: new image
    """
    image = np.full(shape=(height, width), fill_value=fill_value, dtype=np.uint8)
    return image

def generate_noise_image(
        width: int = IMG_SHAPE[1], 
        height: int = IMG_SHAPE[0]
) -> np.ndarray:
    """Generate a noise image

    Args:
        width (int, optional): width of the image. Defaults to IMG_SHAPE[1].
        height (int, optional): height of the image. Defaults to IMG_SHAPE[0].

    Returns:
        np.ndarray: new image
    """
    random_image = np.random.randint(0, 255+1, (height, width), dtype=np.uint8)
    return random_image

def generate_picsum_image(
        save_filepath: str,
        width: int = IMG_SHAPE[1], 
        height: int = IMG_SHAPE[0]
) -> None:
    """Generate a picsum image and save it into a local file

    Args:
        height (int): height of the image. Defaults to IMG_SHAPE[0].
        width (int): width of the image. Defaults to IMG_SHAPE[1].
        save_filepath (str): where to save the image
    """
    _picsum_url = PICSUM_URL.format(width=width, height=height)
    download_image(image_url=_picsum_url, save_path=save_filepath)

def generate_from_random_crop_pdf(
        filepath_or_stream: str,
        crop_width_pct: int, 
        crop_height_pct: int,
        resolution: int = IMG_SHAPE[1] * IMG_SHAPE[0],
        page_nb: Optional[int] = None
) -> np.ndarray:
    """
    Generate an image directly from a crop of pdf file without loading the entire
    pdf file into memory.

    Adjust the value of the crop_width_pct and crop_height_pct to adjust the size of 
    the crop to your problem at end.

    Args:
        filepath_or_stream (str): The pdf to crop from.
        resolution (int, optional): The resolution of the image. Defaults to
            IMG_SHAPE[1] * IMG_SHAPE[0].
        page_nb (Optional[int], optional): The page to crop from. Defaults to None.
        crop_width_pct (int, optional): The width of the crop in percentage.
        crop_height_pct (int, optional): The height of the crop in percentage.

    Returns:
        np.ndarray: The cropped image.
    """
    # random crop location
    x0 = random.uniform(0, 1 - crop_width_pct)
    y0 = random.uniform(0, 1 - crop_height_pct)
    x1 = x0 + crop_width_pct
    y1 = y0 + crop_height_pct

    image = read_pdf_to_images(
        filepath_or_stream=filepath_or_stream,
        page_nb=page_nb,
        resolution=resolution,
        clip_pct=pymupdf.Rect(x0=x0, y0=y0, x1=x1, y1=y1)
    )[0]

    return image