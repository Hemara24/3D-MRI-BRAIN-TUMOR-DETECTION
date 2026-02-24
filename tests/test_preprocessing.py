"""Tests for the MRI preprocessing pipeline."""

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from src.preprocessing import (
    denoise_3t_mri,
    load_mri,
    normalize_intensity,
    preprocess_3t_mri,
    skull_strip,
)


def _make_nifti(shape=(32, 32, 32), low=0.0, high=1000.0) -> str:
    """Write a synthetic NIfTI file and return its path."""
    rng = np.random.default_rng(42)
    data = rng.uniform(low, high, shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    fd, path = tempfile.mkstemp(suffix=".nii")
    os.close(fd)
    nib.save(img, path)
    return path


class TestLoadMri:
    def test_loads_valid_file(self):
        path = _make_nifti()
        try:
            volume, affine = load_mri(path)
            assert volume.shape == (32, 32, 32)
            assert affine.shape == (4, 4)
        finally:
            os.unlink(path)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_mri("/nonexistent/path/scan.nii")

    def test_raises_on_invalid_file(self):
        fd, path = tempfile.mkstemp(suffix=".nii")
        os.close(fd)
        with open(path, "w") as f:
            f.write("not a nifti file")
        try:
            with pytest.raises(ValueError):
                load_mri(path)
        finally:
            os.unlink(path)


class TestNormalizeIntensity:
    def test_output_range(self):
        vol = np.array([[[0.0, 500.0, 1000.0]]], dtype=np.float32)
        normed = normalize_intensity(vol)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

    def test_constant_volume_returns_zeros(self):
        vol = np.ones((4, 4, 4), dtype=np.float32) * 5.0
        normed = normalize_intensity(vol)
        np.testing.assert_array_equal(normed, np.zeros_like(normed))

    def test_preserves_shape(self):
        vol = np.random.rand(10, 10, 10).astype(np.float32)
        normed = normalize_intensity(vol)
        assert normed.shape == vol.shape


class TestDenoise3tMri:
    def test_output_within_range(self):
        vol = np.random.rand(16, 16, 16).astype(np.float32)
        denoised = denoise_3t_mri(vol, sigma=1.0)
        assert denoised.min() >= 0.0
        assert denoised.max() <= 1.0

    def test_output_dtype_float32(self):
        vol = np.random.rand(8, 8, 8).astype(np.float64)
        denoised = denoise_3t_mri(vol)
        assert denoised.dtype == np.float32

    def test_preserves_shape(self):
        vol = np.random.rand(12, 12, 12).astype(np.float32)
        denoised = denoise_3t_mri(vol)
        assert denoised.shape == vol.shape


class TestSkullStrip:
    def test_zeros_out_background(self):
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        # Only the central voxels are "brain"
        vol[3:7, 3:7, 3:7] = 0.8
        stripped = skull_strip(vol, threshold=0.1)
        assert stripped[0, 0, 0] == 0.0  # corner is background

    def test_preserves_shape(self):
        vol = np.random.rand(8, 8, 8).astype(np.float32)
        stripped = skull_strip(vol)
        assert stripped.shape == vol.shape

    def test_output_dtype_float32(self):
        vol = np.random.rand(8, 8, 8).astype(np.float32)
        stripped = skull_strip(vol)
        assert stripped.dtype == np.float32


class TestPreprocess3tMri:
    def test_full_pipeline_output_range(self):
        path = _make_nifti()
        try:
            volume, affine = preprocess_3t_mri(path)
            assert volume.min() >= 0.0
            assert volume.max() <= 1.0
        finally:
            os.unlink(path)

    def test_full_pipeline_shape(self):
        path = _make_nifti(shape=(20, 20, 20))
        try:
            volume, affine = preprocess_3t_mri(path)
            assert volume.shape == (20, 20, 20)
        finally:
            os.unlink(path)

    def test_full_pipeline_raises_on_missing(self):
        with pytest.raises(FileNotFoundError):
            preprocess_3t_mri("/no/such/file.nii.gz")
