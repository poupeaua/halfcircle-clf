"""
Code to generate half circle images
"""

import random
import numpy as np
import cv2

from src.data.config import IMG_SHAPE, MAX_RADIUS, MIN_RADIUS, \
    MIN_THICKNESS, MAX_THICKNESS
from src.data.utils import download_image


def generate_half_circle_image(
        width: int = IMG_SHAPE[1], 
        height: int = IMG_SHAPE[0],
        min_radius: int = MIN_RADIUS,
        max_radius: int = MAX_RADIUS,
        min_thickness: int = MIN_THICKNESS,
        max_thickness: int = MAX_THICKNESS,
        x_center_offset_pct: int = 0.2,
        y_center_offset_pct: int = 0.2,
        min_angle: int = -15,
        max_angle: int = 15
) -> np.ndarray:
    """
    Generates a black image of given width and height with either:
    - A white half-circle
    - Random noise
    
    :param width: Width of the image
    :param height: Height of the image
    :return: Generated image
    """
    # Create a white image
    image = np.full(shape=(height, width), fill_value=255, dtype=np.uint8)

    # center of the half circle configuration
    width_offset = int(width * random.uniform(-x_center_offset_pct, x_center_offset_pct))
    height_offset = int(height * random.uniform(-y_center_offset_pct, y_center_offset_pct))
    center = (width // 2 + width_offset, height // 2 + height_offset)

    # radius configuration
    absolute_limit_radius = min(width, height) // 2
    chose_radius = random.randint(min_radius, max_radius)
    radius = min(absolute_limit_radius, chose_radius)

    # angle configuration
    start_angle = random.randint(min_angle, max_angle) + 180
    end_angle = start_angle + 180

    # thickness configuration
    thickness = random.randint(min_thickness, max_thickness)

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

def generate_filled_image(height: int, width: int, fill_value: int):
    image = np.full(shape=(height, width), fill_value=fill_value, dtype=np.uint8)
    return image

def generate_noise_image(height: int, width: int):
    random_image = np.random.randint(0, 255+1, (height, width), dtype=np.uint8)
    return random_image

def generate_picsum_image(height: int, width: int, save_filepath: str):
    PICSUM_URL = f"https://picsum.photos/{width}/{height}?grayscale"
    download_image(image_url=PICSUM_URL, save_path=save_filepath)

def generate_from_pdf_crop():
    #TODO
    pass