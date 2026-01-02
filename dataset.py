import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HurricaneDamageDataset(Dataset):
    def __init__(self, file_paths, labels, transforms=None):
        """
        Args:
            file_paths (list): List of paths to images.
            labels (list): List of labels (0 or 1).
            transforms (A.Compose): Albumentations transforms.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Read Image
        image = cv2.imread(img_path)
        if image is None:
            # Return a dummy tensor if image is corrupt, or raise error
            # For robustness, let's raise error so user knows
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(phase='train', image_size=128):
    if phase == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
