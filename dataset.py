from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
import numpy as np

class BreastCancerDataset(Dataset):

    def __init__(self, root_dir: str, split: str):
        self.root_dir = Path(root_dir)
        self.split = split

        self.split_dir = self.root_dir / split

        if not self.split_dir.exists():
            raise ValueError(f"Split directory was not found {self.split_dir}")

        self.samples = []

        class_to_label = {
            "benign": 0,
            "malignant": 1
        }

        for class_name, label in class_to_label.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Class directory not found: {class_dir}")
            
            for img_path in class_dir.iterdir():
                if img_path.is_file():
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        y = torch.tensor(label, dtype=torch.long)
        return x, y