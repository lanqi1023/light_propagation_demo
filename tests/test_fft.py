"""Tests for py.fft module."""
import numpy as np
from lightprop.fft import fft2d, fftshift, ifftshift, fft_self_test


def test_fft_self_test_passes():
    results = fft_self_test()
    assert results["1d_impulse_max_err"] < 1e-10
    assert results["2d_roundtrip_max_err"] < 1e-12
    assert results["fftshift_odd_max_err"] < 1e-12
    assert results["fftshift_even_max_err"] < 1e-12


def test_fftshift_ifftshift_roundtrip():
    for nx, ny in [(5, 5), (4, 4), (7, 9), (256, 256)]:
        arr = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
        arr = arr.astype(np.complex128)
        orig = arr.copy()
        arr = fftshift(arr)
        arr = ifftshift(arr)
        np.testing.assert_allclose(arr, orig, atol=1e-14)


def test_fft2d_impulse():
    n = 8
    arr = np.zeros((1, n), dtype=np.complex128)
    arr[0, 0] = 1.0
    fwd = fft2d(arr)
    expected = np.ones_like(fwd)
    np.testing.assert_allclose(fwd, expected, atol=1e-14)


def test_fft2d_roundtrip():
    nx, ny = 16, 16
    orig = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
    orig = orig.astype(np.complex128)
    fwd = fft2d(orig)
    inv = fft2d(fwd, inverse=True)
    np.testing.assert_allclose(inv, orig, atol=1e-13)
