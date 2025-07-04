import os
import random
from typing import List, Tuple, Dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageNetDataset(Dataset):
    """
    A PyTorch Dataset for loading ImageNet validation data and assigning binary labels (e.g., 0 for cat, 1 for dog).
    """
    def __init__(
        self,
        validation_path: str,
        class_mapping_file: str,
        transform=None,
        specific_classes: Dict[str, int] = None,  # Map class names to binary labels (e.g., {"Labrador retriever": 1, "Persian cat": 0})
        fraction_per_class: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize the dataset.

        Args:
            validation_path (str): Path to ImageNet validation set
            class_mapping_file (str): Path to synset words mapping file
            transform: Optional transforms to apply to images
            specific_classes (Dict[str, int]): Dictionary mapping class names to binary labels (e.g., {"dog_class": 1, "cat_class": 0})
            fraction_per_class (float): Fraction of files to include per class (default: 1.0, i.e., all files)
            seed (int): Random seed for class sampling
        """
        if transform == "default":
            transform = get_default_transforms()
        self.transform = transform

        # Convert class names to lowercase for consistency
        self.specific_classes = {k.lower(): v for k, v in specific_classes.items()} if specific_classes else None

        # Load class mappings (WordNet IDs ↔ Class Names)
        self.wnids, self.class_names = self._load_class_mapping(class_mapping_file)
        self.wnid_to_class = dict(zip(self.wnids, self.class_names))
        if specific_classes is not None:
            self.wnid_to_idx = self.specific_classes
        else:
            self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}

        # Create the data list (only classes in specific_classes)
        self.data_list = self._create_data_list(validation_path, seed, fraction_per_class)

    def _load_class_mapping(self, mapping_file: str) -> Tuple[List[str], List[str]]:
        """Load mapping between WordNet IDs and class names."""
        wnids = []
        class_names = []

        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    wnid, class_name = parts
                    class_name = class_name.split(', ')[0].strip().lower()
                    if self.specific_classes is None or wnid in self.specific_classes.keys():
                        class_names.append(class_name)
                        wnids.append(wnid)

        return wnids, class_names

    def _create_data_list(
        self,
        validation_path: str,
        seed: int,
        fraction_per_class: float = 1.0,
    ) -> List[Tuple[str, int]]:
        """
        Create list of (image_path, label) tuples with optional subsampling of files.

        Args:
            validation_path (str): Path to the validation dataset.
            seed (int): Random seed for deterministic sampling.
            fraction_per_class (float): Fraction of files to include per class (default: 1.0, i.e., all files).

        Returns:
            List[Tuple[str, int]]: List of (image_path, label) tuples.
        """
        random.seed(seed)
        data_list = []

        for wnid, label in self.wnid_to_idx.items():
            class_path = os.path.join(validation_path, wnid)
            if not os.path.exists(class_path):
                print(f"Warning: Class path {class_path} does not exist. Skipping.")
                continue

            # Get all files in the class directory and subsample
            all_files = os.listdir(class_path)
            if fraction_per_class < 1.0:
                num_files_to_sample = max(1, int(len(all_files) * fraction_per_class))
                sampled_files = random.sample(all_files, num_files_to_sample)
            else:
                sampled_files = all_files  # Use all files if fraction is 1.0

            # Create (file_path, label) tuples
            for file_name in sampled_files:
                file_path = os.path.join(class_path, file_name)
                data_list.append((file_path, label))

        return data_list

    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        """
        Get an image and its binary label.

        Args:
            idx (int): Index of the data item

        Returns:
            tuple: (image, label, image_path) where image is the transformed PIL Image
                   and label is integer representing the classlabel
        """
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


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


if __name__ == "__main__":
    VALIDATION_PATH = "/common/datasets/ImageNet_ILSVRC2012/val"
    CLASS_MAPPING_FILE = "/common/datasets/ImageNet_ILSVRC2012/synset_words.txt"

    # Define specific classes and their binary labels
    specific_classes = {
        "Labrador retriever": 1,  # Dog → 1
        "Golden retriever": 1,   # Dog → 1
        "Persian cat": 0,        # Cat → 0
        "Siamese cat": 0         # Cat → 0
    }

    dataset = ImageNetDataset(
        validation_path=VALIDATION_PATH,
        class_mapping_file=CLASS_MAPPING_FILE,
        transform=get_default_transforms(),
        specific_classes=specific_classes
    )

    # Test the dataset
    for i in range(5):
        image, label, path = dataset[i]
        print(f"Image shape: {image.shape}, Label: {label}, Path: {path}")