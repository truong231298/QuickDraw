from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class QuickDrawDataset(Dataset):
    """Load per-class Quick, Draw! NPZ archives as normalized grayscale images."""

    def __init__(self, root: str | Path, classes: list[str], max_per_class: int | None = None):
        self.images: list[np.ndarray] = []
        self.labels: list[int] = []
        root = Path(root)
        for label, name in enumerate(classes):
            filename = name.replace(" ", "_")
            path = root / f"{filename}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing class archive: {path}")
            with np.load(path, mmap_mode="r") as archive:
                key = "images" if "images" in archive else "data" if "data" in archive else None
                if key is None:
                    raise ValueError(f"{path} must contain an 'images' or 'data' array")
                arr = archive[key]
                if arr.ndim == 2 and arr.shape[1] == 784:
                    arr = arr.reshape(-1, 28, 28)
                if arr.ndim != 3 or tuple(arr.shape[1:]) != (28, 28):
                    raise ValueError(f"{path} expected N×784 or N×28×28, got {arr.shape}")
                if max_per_class:
                    arr = arr[:max_per_class]
                # Preserve source scale: floating archives are commonly already
                # normalized to [0, 1], while integer archives use [0, 255].
                self.images.extend(np.asarray(arr, dtype=np.float32))
                self.labels.extend([label] * len(arr))
        if not self.images:
            raise ValueError("Dataset contains no images")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        if image.max(initial=0) > 1.0:
            image = image / 255.0
        return torch.from_numpy(image[None, ...]), self.labels[index]
