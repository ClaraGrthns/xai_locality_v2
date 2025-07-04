import os
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
def get_default_transforms():
    """Get the default ImageNet transforms."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class CatsVsDogsDataset(Dataset):
    def __init__(self, root_dir: str = "/home/grotehans/xai_locality/data/cats_vs_dogs/train", transform=None):
        """
        Custom Dataset for Kaggle Cats vs. Dogs.

        Args:
            root_dir (str): Path to the dataset directory (e.g., "train/").
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.root_dir = root_dir
        if transform == "default":
            transform = get_default_transforms()
        self.transform = transform
        self.data = []

        # Load all images and their corresponding labels
        for filename in os.listdir(root_dir):
            if filename.startswith("cat"):
                label = 0  # Cat
            elif filename.startswith("dog"):
                label = 1  # Dog
            else:
                label= 0.5  # test images

            img_path = os.path.join(root_dir, filename)
            self.data.append((img_path, label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label.
        Args:
            idx (int): Index of the data item.
        Returns:
            tuple: (image, label), where image is a transformed tensor and label is an integer.
        """
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")  # Open image as RGB
        
        if self.transform:
            image = self.transform(image)

        return image, label, img_path
    
