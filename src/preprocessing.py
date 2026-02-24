"""
Preprocessing pipeline for low-quality 3T MRI volumes.

Steps applied in order:
1. Load NIfTI volume (.nii / .nii.gz).
2. Intensity normalisation to [0, 1].
3. Gaussian denoising tuned for 3T scanner noise.
4. Simple threshold-based skull stripping.
"""

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


def load_mri(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI MRI file and return the volume and affine matrix.

    Parameters
    ----------
    filepath : str
        Path to a .nii or .nii.gz file.

    Returns
    -------
    volume : ndarray, shape (X, Y, Z)
        Raw voxel data as float32.
    affine : ndarray, shape (4, 4)
        Affine transformation matrix from the NIfTI header.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file cannot be read as a NIfTI image.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"MRI file not found: {filepath}")
    try:
        img = nib.load(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot load NIfTI file '{filepath}': {exc}") from exc
    volume = img.get_fdata(dtype=np.float32)
    return volume, img.affine


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Rescale voxel intensities to the range [0, 1].

    Parameters
    ----------
    volume : ndarray
        Input MRI volume (arbitrary dtype).

    Returns
    -------
    ndarray, float32
        Normalised volume in [0, 1].
    """
    volume = volume.astype(np.float32)
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume)
    return (volume - vmin) / (vmax - vmin)


def denoise_3t_mri(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to suppress thermal noise typical of 3T scanners.

    Parameters
    ----------
    volume : ndarray
        Normalised MRI volume.
    sigma : float
        Standard deviation for the Gaussian kernel (default 1.0).

    Returns
    -------
    ndarray, float32
        Denoised volume, clipped to [0, 1].
    """
    denoised = ndimage.gaussian_filter(volume.astype(np.float32), sigma=sigma)
    return np.clip(denoised, 0.0, 1.0)


def skull_strip(volume: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Remove non-brain tissue using intensity thresholding and morphology.

    Parameters
    ----------
    volume : ndarray
        Denoised, normalised MRI volume.
    threshold : float
        Intensity threshold for the initial brain mask (default 0.1).

    Returns
    -------
    ndarray, float32
        Brain-masked volume with non-brain voxels zeroed out.
    """
    mask = volume > threshold
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=2)
    mask = ndimage.binary_dilation(mask, iterations=2)
    return (volume * mask).astype(np.float32)


def preprocess_3t_mri(
    filepath: str,
    denoise_sigma: float = 1.0,
    skull_strip_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline for a single 3T MRI scan.

    Applies load → normalise → denoise → skull-strip in sequence.

    Parameters
    ----------
    filepath : str
        Path to the NIfTI file.
    denoise_sigma : float
        Gaussian sigma for denoising (default 1.0).
    skull_strip_threshold : float
        Intensity threshold for skull stripping (default 0.1).

    Returns
    -------
    volume : ndarray, shape (X, Y, Z), float32
        Preprocessed MRI volume ready for graph construction.
    affine : ndarray, shape (4, 4)
        Affine matrix from the original NIfTI header.
    """
    volume, affine = load_mri(filepath)
    volume = normalize_intensity(volume)
    volume = denoise_3t_mri(volume, sigma=denoise_sigma)
    volume = skull_strip(volume, threshold=skull_strip_threshold)
    return volume, affine
