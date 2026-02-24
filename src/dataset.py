"""
PyTorch Dataset for 3T MRI brain tumour detection.

Expects a CSV manifest file with columns:
    filepath  – path to a NIfTI (.nii / .nii.gz) file
    label     – integer class label (0 = no tumour, 1 = tumour)

Each item returned is a ``torch_geometric.data.Data`` graph built by
the preprocessing and graph-construction pipelines.
"""

import csv
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch_geometric.data import Dataset, Data

from .graph_construction import build_graph_from_volume
from .preprocessing import preprocess_3t_mri


class MRITumorDataset(Dataset):
    """Dataset of graph-encoded 3T MRI scans for tumour classification.

    Parameters
    ----------
    manifest_csv : str
        Path to a CSV file with ``filepath`` and ``label`` columns.
    n_segments : int
        Number of SLIC supervoxel segments per scan (default 500).
    k_neighbors : int
        Number of k-NN neighbours for edge construction (default 10).
    transform : callable, optional
        Optional transform applied to each ``Data`` object.
    denoise_sigma : float
        Gaussian sigma for 3T denoising in preprocessing (default 1.0).
    skull_strip_threshold : float
        Intensity threshold for skull stripping (default 0.1).
    """

    def __init__(
        self,
        manifest_csv: str,
        n_segments: int = 500,
        k_neighbors: int = 10,
        transform: Optional[Callable] = None,
        denoise_sigma: float = 1.0,
        skull_strip_threshold: float = 0.1,
    ) -> None:
        super().__init__(transform=transform)
        self.n_segments = n_segments
        self.k_neighbors = k_neighbors
        self.denoise_sigma = denoise_sigma
        self.skull_strip_threshold = skull_strip_threshold
        self._samples: List[Tuple[str, int]] = self._load_manifest(manifest_csv)

    @staticmethod
    def _load_manifest(csv_path: str) -> List[Tuple[str, int]]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")
        samples: List[Tuple[str, int]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                samples.append((row["filepath"].strip(), int(row["label"])))
        return samples

    def len(self) -> int:  # noqa: D102
        return len(self._samples)

    def get(self, idx: int) -> Data:  # noqa: D102
        filepath, label = self._samples[idx]
        volume, _ = preprocess_3t_mri(
            filepath,
            denoise_sigma=self.denoise_sigma,
            skull_strip_threshold=self.skull_strip_threshold,
        )
        return build_graph_from_volume(
            volume,
            n_segments=self.n_segments,
            k_neighbors=self.k_neighbors,
            label=label,
        )
