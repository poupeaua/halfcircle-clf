"""
Pytorch dataset class
"""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HalfCircleBinaryClfDataset(Dataset):
    def __init__(self, images_filepaths: list[str], transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = Image.open(image_filepath)
        if "/hc/" not in image_filepath:
            label = 0.0
        else:
            label = 1.0
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label