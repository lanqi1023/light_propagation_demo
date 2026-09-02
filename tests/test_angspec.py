"""Tests for py.angspec module."""
import numpy as np
from lightprop.angspec import propagate


def test_basic_propagation_shape():
    Nx, Ny = 64, 64
    dx, dy = 10.0, 10.0
    U0 = np.zeros((Nx, Ny), dtype=np.complex128)
    U0[32, 32] = 1.0
    Uz = propagate(U0, Nx, Ny, dx, dy, 532.0, 100.0)
    assert Uz.shape == (Nx, Ny)
    assert Uz.dtype == np.complex128


def test_energy_conservation():
    """For lossless propagation with no band-limiting, total intensity should be conserved."""
    Nx, Ny = 64, 64
    dx, dy = 10.0, 10.0
    U0 = np.zeros((Nx, Ny), dtype=np.complex128)
    U0[32, 32] = 1.0
    Uz = propagate(U0, Nx, Ny, dx, dy, 532.0, 1.0, padding=True, band_limited=False)
    E_in = np.sum(np.abs(U0) ** 2)
    E_out = np.sum(np.abs(Uz) ** 2)
    # FFT numerical precision tolerance
    np.testing.assert_allclose(E_out, E_in, rtol=1e-4)


def test_non_square_pixels():
    """With dx != dy, propagation should work without axis confusion."""
    Nx, Ny = 64, 64
    dx, dy = 10.0, 20.0
    U0 = np.zeros((Nx, Ny), dtype=np.complex128)
    U0[32, 32] = 1.0
    Uz = propagate(U0, Nx, Ny, dx, dy, 532.0, 100.0)
    assert Uz.shape == (Nx, Ny)
    assert np.isfinite(Uz).all()
    assert not np.allclose(Uz, 0.0)


def test_band_limiting_supports_partial_energy():
    """Band-limiting at moderate dz should preserve some energy for low-freq content."""
    Nx, Ny = 64, 64
    dx, dy = 10.0, 10.0
    # Low-frequency aperture (wide slit)
    U0 = np.zeros((Nx, Ny), dtype=np.complex128)
    U0[:, 24:40] = 1.0
    Uz = propagate(U0, Nx, Ny, dx, dy, 532.0, 100.0, padding=True, band_limited=True)
    E_out = np.sum(np.abs(Uz) ** 2)
    assert E_out > 0


def test_padding_conserves_energy():
    """Both padded and non-padded should conserve energy (without band-limiting)."""
    Nx, Ny = 32, 32
    dx, dy = 10.0, 10.0
    U0 = np.zeros((Nx, Ny), dtype=np.complex128)
    U0[16, 16] = 1.0
    E_in = np.sum(np.abs(U0) ** 2)
    Uz_pad = propagate(U0.copy(), Nx, Ny, dx, dy, 532.0, 1.0, padding=True, band_limited=False)
    Uz_nopad = propagate(U0.copy(), Nx, Ny, dx, dy, 532.0, 1.0, padding=False, band_limited=False)
    E_pad = np.sum(np.abs(Uz_pad) ** 2)
    E_nopad = np.sum(np.abs(Uz_nopad) ** 2)
    # Zero-padding + center extraction introduces tiny interpolation error (~0.05%)
    np.testing.assert_allclose(E_pad, E_in, rtol=1e-3)
    np.testing.assert_allclose(E_nopad, E_in, rtol=1e-4)


def test_output_is_finite():
    """Propagation result should always be finite."""
    Nx, Ny = 64, 64
    for dx, dy in [(10.0, 10.0), (10.0, 20.0), (5.0, 15.0)]:
        U0 = np.zeros((Nx, Ny), dtype=np.complex128)
        U0[32, 32] = 1.0
        for dz in [1.0, 100.0, 500.0]:
            Uz = propagate(U0.copy(), Nx, Ny, dx, dy, 532.0, dz)
            assert np.isfinite(Uz).all()
