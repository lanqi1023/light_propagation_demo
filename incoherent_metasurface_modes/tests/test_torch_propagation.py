import math

import pytest

torch = pytest.importorskip("torch")

from metasurface_modes.torch_api import (  # noqa: E402
    AngularSpectrumPropagator,
    AreaDetector,
    CartesianGrid,
    CollimatingMetalens,
    IncoherentMetasurfaceModel,
    LambertianLEDArray,
    LambertianLEDArrayConfig,
    MultilayerMetasurface,
    circular_aperture,
    rectangular_aperture,
    tiled_detector_masks,
)


def make_small_model(*, n_layers=1, dtype=torch.float64):
    wavelength_m = 532e-9
    focal_length_m = 1.0e-3
    grid = CartesianGrid(16, 16, 8e-6, 8e-6)
    aperture = circular_aperture(
        grid,
        112e-6,
        dtype=dtype,
    )
    source = LambertianLEDArray(
        LambertianLEDArrayConfig(
            layout_y=1,
            layout_x=2,
            pitch_y_m=20e-6,
            pitch_x_m=20e-6,
            source_to_metasurface_m=focal_length_m,
        ),
        aperture=aperture,
    )
    optical = MultilayerMetasurface(
        grid,
        wavelength_m,
        focal_length_m,
        n_trainable_layers=n_layers,
        propagation_distances_m=tuple([0.4e-3] * (n_layers + 1)),
        aperture=aperture,
        dtype=dtype,
    )
    detector = AreaDetector(tiled_detector_masks(grid, 2, 2, dtype=dtype))
    return IncoherentMetasurfaceModel(source, optical, detector)


def test_angular_spectrum_preserves_power_when_all_sampled_orders_propagate():
    torch.manual_seed(1)
    grid = CartesianGrid(16, 20, 4e-6, 4e-6)
    propagator = AngularSpectrumPropagator(
        grid,
        wavelength_m=532e-9,
        distance_m=1e-3,
        dtype=torch.float64,
    )
    field = torch.randn(grid.shape, dtype=torch.float64) + 1j * torch.randn(
        grid.shape,
        dtype=torch.float64,
    )
    output = propagator(field)
    torch.testing.assert_close(
        output.abs().square().sum(),
        field.abs().square().sum(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_center_led_phase_is_flat_after_exact_collimating_metalens():
    wavelength_m = 532e-9
    focal_length_m = 1e-3
    grid = CartesianGrid(17, 17, 6e-6, 6e-6)
    aperture = circular_aperture(grid, 90e-6, dtype=torch.float64)
    source = LambertianLEDArray(
        LambertianLEDArrayConfig(
            layout_y=1,
            layout_x=1,
            pitch_y_m=10e-6,
            pitch_x_m=10e-6,
            source_to_metasurface_m=focal_length_m,
        ),
        aperture=aperture,
    )
    mode, _ = source.unit_modes(
        grid,
        wavelength_m,
        dtype=torch.float64,
    )
    lens = CollimatingMetalens(
        grid,
        wavelength_m,
        focal_length_m,
        aperture=aperture,
        dtype=torch.float64,
    )
    corrected = lens(mode[0])
    values = corrected[aperture.bool()]
    unit_phasor = values / values.abs()
    reference = unit_phasor.mean()
    reference = reference / reference.abs()
    torch.testing.assert_close(
        unit_phasor,
        reference.expand_as(unit_phasor),
        rtol=1e-9,
        atol=1e-9,
    )


def test_exact_forward_equals_explicit_intensity_matrix_product():
    model = make_small_model(n_layers=1)
    powers = torch.tensor([[0.2, 0.9], [1.0, 0.3]], dtype=torch.float64)
    matrix = model.intensity_matrix(mode_chunk_size=1)
    expected = powers @ matrix.transpose(0, 1)
    actual = model(powers, method="exact", mode_chunk_size=1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_random_phase_estimator_converges_to_exact_incoherent_result():
    model = make_small_model(n_layers=0)
    powers = torch.tensor([0.4, 1.0], dtype=torch.float64)
    exact = model(powers, method="exact")
    generator = torch.Generator().manual_seed(12)
    estimate = model(
        powers,
        method="stochastic",
        n_realizations=4096,
        realization_chunk_size=128,
        mode_chunk_size=2,
        generator=generator,
    )
    torch.testing.assert_close(estimate, exact, rtol=0.06, atol=1e-10)


def test_trainable_phase_receives_finite_gradient():
    model = make_small_model(n_layers=1, dtype=torch.float32)
    powers = torch.tensor([[1.0, 0.3], [0.2, 0.9]], dtype=torch.float32)
    output = model(powers, method="exact", mode_chunk_size=2)
    detector_weights = torch.tensor([1.0, -0.2, 0.4, -0.7])
    loss = (output * detector_weights).sum()
    loss.backward()
    gradient = model.optical_system.trainable_surfaces[0].phase.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(torch.linalg.vector_norm(gradient)) > 0.0


def test_lambertian_collection_is_positive_and_below_unity():
    grid = CartesianGrid(32, 32, 8e-6, 8e-6)
    aperture = circular_aperture(grid, 200e-6, dtype=torch.float64)
    source = LambertianLEDArray(
        LambertianLEDArrayConfig(
            layout_y=1,
            layout_x=1,
            pitch_y_m=10e-6,
            pitch_x_m=10e-6,
            source_to_metasurface_m=1e-3,
            emitter_height_m=4e-6,
            emitter_width_m=4e-6,
            emitter_samples_y=2,
            emitter_samples_x=2,
        ),
        aperture=aperture,
    )
    efficiency = source.collection_efficiencies(
        grid,
        532e-9,
        dtype=torch.float64,
    )[0]
    assert 0.0 < float(efficiency) < 1.0
    # Small-angle circular-aperture estimate: eta ~= NA^2.
    radius = 100e-6
    approximate_na = radius / math.sqrt(radius**2 + (1e-3) ** 2)
    assert float(efficiency) == pytest.approx(approximate_na**2, rel=0.2)


def test_rectangular_aperture_has_requested_sampled_support():
    grid = CartesianGrid(5, 7, 1.0, 1.0)
    aperture = rectangular_aperture(
        grid,
        height_m=3.0,
        width_m=5.0,
        dtype=torch.float64,
    )
    # Centered coordinates are y=-2..2 and x=-3..3, so inclusive half-width
    # tests select 3 rows and 5 columns.
    assert int(aperture.sum()) == 15
    assert torch.all(aperture[1:4, 1:6] == 1)
    assert torch.all(aperture[[0, 4], :] == 0)
