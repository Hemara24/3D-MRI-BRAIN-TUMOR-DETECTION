"""
Supervoxel-based graph construction for 3T MRI volumes.

Each node in the graph represents one SLIC supervoxel; node features
capture regional intensity statistics and morphological properties.
Edges connect spatially adjacent supervoxels via a k-nearest-neighbour
search on supervoxel centroids.
"""

from typing import Optional

import numpy as np
import torch
from scipy.spatial import KDTree
from skimage.measure import regionprops
from skimage.segmentation import slic
from torch_geometric.data import Data


def _region_features(region, volume: np.ndarray) -> list:
    """Extract a fixed-length feature vector from a regionprops object.

    Returns
    -------
    list of float
        [mean_intensity, std_intensity, area,
         major_axis_length, minor_axis_length, solidity]
    """
    # Voxels belonging to this region
    coords = region.coords
    intensities = volume[coords[:, 0], coords[:, 1], coords[:, 2]]

    mean_int = float(intensities.mean())
    std_int = float(intensities.std())
    area = float(region.area)

    # axis lengths / solidity may not exist for very small regions;
    # prefer the non-deprecated attribute names introduced in skimage 0.26
    major = float(
        getattr(region, "axis_major_length", None)
        or getattr(region, "major_axis_length", 0.0)
        or 0.0
    )
    minor = float(
        getattr(region, "axis_minor_length", None)
        or getattr(region, "minor_axis_length", 0.0)
        or 0.0
    )
    solidity = float(getattr(region, "solidity", 1.0) or 1.0)

    return [mean_int, std_int, area, major, minor, solidity]


# Number of features produced by _region_features – used as a public constant
# so downstream code can reference it without hard-coding.
NUM_NODE_FEATURES: int = 6


def build_graph_from_volume(
    volume: np.ndarray,
    n_segments: int = 500,
    compactness: float = 10.0,
    sigma: float = 1.0,
    k_neighbors: int = 10,
    label: Optional[int] = None,
) -> Data:
    """Build a PyG ``Data`` graph from a preprocessed MRI volume.

    Parameters
    ----------
    volume : ndarray, shape (X, Y, Z), float32
        Preprocessed MRI volume (output of the preprocessing pipeline).
    n_segments : int
        Target number of supervoxels (default 500).
    compactness : float
        SLIC compactness parameter (default 10.0).
    sigma : float
        SLIC Gaussian smoothing before segmentation (default 1.0).
    k_neighbors : int
        Number of nearest neighbours used to build edges (default 10).
    label : int or None
        Graph-level class label (0 = no tumor, 1 = tumor).
        When ``None`` the returned ``Data`` object has no ``y`` attribute.

    Returns
    -------
    torch_geometric.data.Data
        Graph with node features ``x``, edge connectivity ``edge_index``,
        and optionally a graph label ``y``.
    """
    # SLIC supervoxel segmentation (channel_axis=None for 3-D greyscale)
    segments = slic(
        volume,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        channel_axis=None,
    )

    # regionprops uses 1-indexed labels
    regions = regionprops(segments + 1)

    if len(regions) == 0:
        raise ValueError("SLIC produced zero segments – check input volume.")

    # Node features
    features = np.array(
        [_region_features(r, volume) for r in regions], dtype=np.float32
    )

    # Centroids for k-NN edge construction
    centroids = np.array([r.centroid for r in regions], dtype=np.float32)

    # Build bidirectional k-NN edges
    k = min(k_neighbors, len(centroids) - 1)
    tree = KDTree(centroids)
    edge_list = []
    for i, centroid in enumerate(centroids):
        _, indices = tree.query(centroid, k=k + 1)  # +1 because self is included
        for j in indices[1:]:  # skip self-loop
            edge_list.append([i, int(j)])
            edge_list.append([int(j), i])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    return data
