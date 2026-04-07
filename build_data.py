import os
from typing import Optional, Sequence, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


class ImageSpectrumDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        spectrum_file: str,
        transform=None,
        allowed_extensions: Optional[Sequence[str]] = None,
    ):
        self.spectrum_data = pd.read_excel(spectrum_file)
        self.image_dir = image_dir
        self.transform = transform
        self.allowed_extensions = allowed_extensions or [".png", ".bmp"]

        self.image_names = self.spectrum_data.iloc[:, 0].astype(str).tolist()
        self.spectra = self.spectrum_data.iloc[:, 1:].values.astype("float32")

    def __len__(self):
        return len(self.image_names)

    def _resolve_image_path(self, base_image_name: str) -> str:
        for ext in self.allowed_extensions:
            candidate = os.path.join(self.image_dir, base_image_name + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"No image found for '{base_image_name}' with extensions {self.allowed_extensions}"
        )

    def __getitem__(self, idx):
        base_image_name = self.image_names[idx]
        img_path = self._resolve_image_path(base_image_name)

        image = Image.open(img_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)

        spectrum = torch.tensor(self.spectra[idx], dtype=torch.float32)
        return image, spectrum



############################
# splits

def make_run_splits(
    dataset_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_size)
    rng.shuffle(indices)

    n_train = int(dataset_size * train_ratio)
    n_val = int(dataset_size * val_ratio)
    n_test = dataset_size - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


# def make_percentage_train_subset(
#     train_indices: np.ndarray,
#     train_percent: float,
#     seed: int,
# ) -> np.ndarray:
#     if not (0 < train_percent <= 1.0):
#         raise ValueError("train_percent must be in (0, 1]")

#     rng = np.random.default_rng(seed)
#     shuffled = train_indices.copy()
#     rng.shuffle(shuffled)

#     subset_size = max(1, int(len(shuffled) * train_percent))
#     return shuffled[:subset_size]

# nested subsets version:  60% subset ⊂ 80% subset ⊂ 100% subset
def make_percentage_train_subset(
    train_indices: np.ndarray,
    train_percent: float,
    seed: int,
) -> np.ndarray:
    if not (0 < train_percent <= 1.0):
        raise ValueError("train_percent must be in (0, 1]")

    rng = np.random.default_rng(seed)
    shuffled = train_indices.copy()
    rng.shuffle(shuffled)

    subset_size = max(1, int(len(shuffled) * train_percent))
    return shuffled[:subset_size]


def build_subsets_for_experiment(
    dataset,
    splits: Dict[str, np.ndarray],
    train_percent: float,
    subset_seed: int,
) -> Tuple[Subset, Subset, Subset, np.ndarray]:
    train_subset_idx = make_percentage_train_subset(
        train_indices=splits["train"],
        train_percent=train_percent,
        seed=subset_seed,
    )

    train_dataset = Subset(dataset, train_subset_idx.tolist())
    val_dataset = Subset(dataset, splits["val"].tolist())
    test_dataset = Subset(dataset, splits["test"].tolist())

    return train_dataset, val_dataset, test_dataset, train_subset_idx    