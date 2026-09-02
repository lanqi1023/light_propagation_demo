"""Tests for py.aperture module."""
import numpy as np
from lightprop.aperture import generate, _grids


def test_grids():
    xs, ys = _grids(5, 7, 10.0, 20.0)
    assert len(xs) == 7
    assert len(ys) == 5
    assert np.isclose(xs[3], 0.0)
    assert np.isclose(ys[2], 0.0)


def test_slit():
    U = generate("slit", 64, 64, 10.0, 10.0, {"width": 100.0})
    assert U.shape == (64, 64)
    assert U.dtype == np.complex128
    # Center inside slit
    assert U[32, 32] == 1.0
    # Edges outside
    assert U[32, 0] == 0.0
    assert U[32, 63] == 0.0


def test_double_slit():
    U = generate("doubleSlit", 64, 64, 10.0, 10.0,
                 {"width": 50.0, "separation": 200.0})
    assert U[32, 32] == 0.0  # center between slits
    # Slit centers approx at +/- separation/2
    # separation=200, dx=10, center=31.5 -> +/-10 -> indices 21 and 41
    assert U[32, 21] == 1.0 or U[32, 22] == 1.0
    assert U[32, 41] == 1.0 or U[32, 40] == 1.0


def test_circle():
    U = generate("circle", 64, 64, 10.0, 10.0, {"radius": 100.0})
    assert U[32, 32] == 1.0  # center
    assert U[0, 0] == 0.0    # corner


def test_rectangle():
    U = generate("rectangle", 64, 64, 10.0, 10.0,
                 {"width": 100.0, "height": 200.0})
    assert U[32, 32] == 1.0
    assert U[32, 0] == 0.0


def test_free_space():
    U = generate("free", 32, 32, 10.0, 10.0, {})
    assert np.all(U == 1.0)


def test_gaussian_envelope():
    U = generate("slit", 64, 64, 10.0, 10.0, {"width": 500.0}, w0=50.0)
    center = np.abs(U[32, 32])
    corner = np.abs(U[0, 0])
    assert center > corner


def test_plane_wave_skip_envelope():
    U1 = generate("slit", 64, 64, 10.0, 10.0, {"width": 500.0}, w0=100.0)
    U2 = generate("slit", 64, 64, 10.0, 10.0, {"width": 500.0}, w0=200.0)
    np.testing.assert_array_equal(U1, U2)
