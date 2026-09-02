import numpy as np

from metasurface_modes import (
    AngularModeArrayConfig,
    gram_diagnostics,
    gram_matrix_from_modes,
    helstrom_intensity_columns,
    mode_overlap_for_offset,
    output_total_variation_bound,
    summarize_regular_array,
    target_pair_feasibility,
)


def orthogonal_grid_config(layout_x=8, layout_y=6):
    wavelength_nm = 500.0
    focal_length_mm = 10.0
    pitch_um = 10.0
    # D * pitch / (lambda * f) = 1: all nonzero integer offsets lie
    # on zeros of the rectangular-pupil sinc overlap.
    pupil_mm = (
        (wavelength_nm / 1000.0)
        * (focal_length_mm * 1000.0)
        / pitch_um
        / 1000.0
    )
    return AngularModeArrayConfig(
        layout_x=layout_x,
        layout_y=layout_y,
        led_pitch_x_um=pitch_um,
        led_pitch_y_um=pitch_um,
        wavelength_nm=wavelength_nm,
        focal_length_mm=focal_length_mm,
        pupil_shape="rectangle",
        pupil_width_mm=pupil_mm,
        pupil_height_mm=pupil_mm,
    )


def test_fourier_grid_modes_are_orthogonal():
    config = orthogonal_grid_config()
    assert abs(float(mode_overlap_for_offset(config, 1, 0))) < 1e-14
    assert abs(float(mode_overlap_for_offset(config, 0, 1))) < 1e-14
    assert abs(float(mode_overlap_for_offset(config, 3, 2))) < 1e-14


def test_effective_rank_equals_mode_count_on_orthogonal_grid():
    config = orthogonal_grid_config(layout_x=9, layout_y=7)
    summary = summarize_regular_array(config)
    np.testing.assert_allclose(summary.effective_rank, config.n_led, rtol=1e-14)
    np.testing.assert_allclose(summary.effective_rank_fraction, 1.0, rtol=1e-14)
    assert summary.max_off_diagonal_overlap < 1e-14


def test_small_pupil_makes_modes_nonorthogonal_and_reduces_rank():
    base = orthogonal_grid_config(layout_x=9, layout_y=7)
    config = AngularModeArrayConfig(
        **{
            **base.__dict__,
            "pupil_width_mm": base.pupil_width_mm * 0.2,
            "pupil_height_mm": base.pupil_height_mm * 0.2,
        }
    )
    summary = summarize_regular_array(config)
    assert summary.max_off_diagonal_overlap > 0.9
    assert summary.effective_rank < 0.2 * config.n_led


def test_distinguishability_bound_endpoints():
    np.testing.assert_allclose(output_total_variation_bound(0.0), 1.0)
    np.testing.assert_allclose(output_total_variation_bound(1.0), 0.0)


def test_optimal_lossless_measurement_saturates_bound():
    for overlap in (0.0, 0.2, 0.7, 0.99, 1.0):
        column_a, column_b = helstrom_intensity_columns(overlap)
        result = target_pair_feasibility(overlap, column_a, column_b)
        np.testing.assert_allclose(
            result["total_variation"],
            result["max_total_variation"],
            atol=1e-12,
        )
        assert result["feasible"]


def test_disjoint_outputs_require_orthogonal_inputs():
    column_a = np.array([1.0, 0.0])
    column_b = np.array([0.0, 1.0])
    assert target_pair_feasibility(0.0, column_a, column_b)["feasible"]
    assert not target_pair_feasibility(0.1, column_a, column_b)["feasible"]


def test_numerical_gram_matrix_detects_duplicate_modes():
    modes = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    gram = gram_matrix_from_modes(modes)
    stats = gram_diagnostics(gram)
    assert stats["numerical_rank"] == 2
    assert stats["max_off_diagonal_overlap"] == 1.0
    assert stats["effective_rank"] < 3.0


def test_circle_overlap_is_one_at_zero_offset():
    config = AngularModeArrayConfig(pupil_shape="circle")
    np.testing.assert_allclose(mode_overlap_for_offset(config, 0, 0), 1.0)

