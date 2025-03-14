"""
Transforms images
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
import random

from src.data.config import IMG_SHAPE, IMG_RANDOM_RESIZE_CROP_SCALE

class RandomDilateTransform:
    def __init__(self, min_kernel=3, max_kernel=7, p=0.5):
        """
        Custom transform to randomly apply morphological dilation with variable strength.
        
        :param min_kernel: Minimum size of the dilation kernel.
        :param max_kernel: Maximum size of the dilation kernel.
        :param p: Probability of applying dilation.
        """
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.p = p
    
    def __call__(self, tensor_image):
        """Applies dilation randomly with a kernel size chosen in the given range."""
        if random.random() < self.p:
            tensor_image = 1 - tensor_image
            kernel_size = random.randint(self.min_kernel, self.max_kernel)
            if kernel_size % 2 == 0:
                kernel_size += 1 # needed to ensure same output size
            padding = (kernel_size - 1) // 2  # Ensure same output size
            dilated_tensor = torch.nn.functional.max_pool2d(tensor_image, kernel_size, stride=1, padding=padding)
            dilated_tensor = 1 - dilated_tensor
            return dilated_tensor
        return tensor_image

# Transform pipeline for training grayscale images (with augmentations)
TRAIN_TRANSFORMS = T.Compose([
    T.Grayscale(num_output_channels=1),  # Convert to grayscale
    T.Resize(IMG_SHAPE),  # Resize to a fixed size
    T.RandomHorizontalFlip(p=0.5),  # Flip horizontally with 50% probability
    T.RandomRotation(10, fill=255),  # Rotate by ±15 degrees
    T.RandomAffine(degrees=0, translate=(0.2, 0.2), fill=255),
    T.ToTensor(),  # Convert image to tensor
    T.GaussianNoise(sigma=0.01),
    RandomDilateTransform(min_kernel=3, max_kernel=7, p=0.5),  # Random dilation strength
    T.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])

# Transform pipeline for testing/validation grayscale images (no augmentations)
TEST_TRANSFORMS = T.Compose([
    T.Grayscale(num_output_channels=1),  # Convert to grayscale
    T.Resize(IMG_SHAPE),  # Resize to a fixed size
    T.ToTensor(),  # Convert image to tensor
    T.ConvertImageDtype(torch.float32),  # Convert image to float
    T.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])
