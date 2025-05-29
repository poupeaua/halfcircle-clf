"""
Transforms images
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
import random

from src.data.config import IMG_SHAPE

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
            dilated_tensor = torch.nn.functional.max_pool2d(
                tensor_image, kernel_size, stride=1, padding=padding
            )
            dilated_tensor = 1 - dilated_tensor
            return dilated_tensor
        return tensor_image
    

class RandomThreshold:
    def __init__(self, p=0.5, min_threshold=0.25, max_threshold=0.75):
        self.p = p
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        assert self.min_threshold < self.max_threshold
        assert self.min_threshold >= 0 and self.max_threshold <= 1
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Grayscale image to be thresholded.
        Returns:
            Tensor: Thresholded binary image.
        """
        if random.random() > self.p:
            return img
        
        if isinstance(img, torch.Tensor):
            img = img.clone()  # Avoid modifying the original image
        else:
            img = T.ToTensor()(img)
        
        strength = torch.rand(1).item()  # Random value between 0 and 1
        threshold = self.min_threshold + \
            strength * (self.max_threshold - self.min_threshold)
        return (img > threshold).float()  # Apply threshold and return binary image

class ToTensorNew:

    def __call__(self, img):
        """
        Convert a PIL Image or numpy.ndarray to a tensor.
        
        Args:
            img (PIL Image or numpy.ndarray): Image to be converted.
        
        Returns:
            Tensor: Converted image as a tensor.
        """
        if isinstance(img, torch.Tensor):
            return img
        return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(img)

# Transform pipeline for training grayscale images (with augmentations)
TRAIN_TRANSFORMS = T.Compose([
    T.Grayscale(num_output_channels=1),  # Convert to grayscale
    T.Resize(IMG_SHAPE),  # Resize to a fixed size
    T.RandomHorizontalFlip(p=0.5),  # Flip horizontally with 50% probability
    T.RandomRotation(10, fill=255),  # Rotate by ±15 degrees
    T.RandomAffine(degrees=0, translate=(0.2, 0.2), fill=255),
    T.ToTensor(),  # Convert image to tensor
    T.GaussianNoise(sigma=0.01),
    RandomThreshold(p=0.5, min_threshold=0.25, max_threshold=0.75), # Random threshold
    RandomDilateTransform(min_kernel=3, max_kernel=7, p=0.5), # Random dilation strength
    T.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])

# Transform pipeline for testing/validation grayscale images (no augmentations)
INFERENCE_TRANSFORMS = T.Compose([
    T.Grayscale(num_output_channels=1),  # Convert to grayscale
    T.Resize(IMG_SHAPE),  # Resize to a fixed size
    T.ToTensor(),  # Convert image to tensor
    T.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])
