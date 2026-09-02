"""Analytic overlap model for a regular MicroLED array and an ideal Fourier lens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import j1

from .diagnostics import ModeOverlapSummary, output_total_variation_bound


PupilShape = Literal["rectangle", "circle"]


@dataclass(frozen=True)
class AngularModeArrayConfig:
    """Physical parameters for a regular LED position-to-angle mapping.

    Lengths are μm except focal length and pupil dimensions, which are mm.
    The default 256×192 arrangement is only a placeholder rectangular layout for
    49,152 physical LEDs; replace it with the real layout before interpretation.
    """

    layout_x: int = 256
    layout_y: int = 192
    led_pitch_x_um: float = 10.0
    led_pitch_y_um: float = 10.0
    wavelength_nm: float = 532.0
    focal_length_mm: float = 20.0
    pupil_shape: PupilShape = "rectangle"
    pupil_width_mm: float = 1.064
    pupil_height_mm: float = 1.064
    pupil_diameter_mm: float = 1.064

    @property
    def n_led(self) -> int:
        return self.layout_x * self.layout_y

    @property
    def wavelength_um(self) -> float:
        return self.wavelength_nm / 1000.0

    @property
    def focal_length_um(self) -> float:
        return self.focal_length_mm * 1000.0


def _validate_config(config: AngularModeArrayConfig) -> None:
    if config.layout_x <= 0 or config.layout_y <= 0:
        raise ValueError("layout dimensions must be positive")
    if config.led_pitch_x_um <= 0 or config.led_pitch_y_um <= 0:
        raise ValueError("LED pitches must be positive")
    if config.wavelength_nm <= 0 or config.focal_length_mm <= 0:
        raise ValueError("wavelength and focal length must be positive")
    if config.pupil_shape == "rectangle":
        if config.pupil_width_mm <= 0 or config.pupil_height_mm <= 0:
            raise ValueError("rectangular pupil dimensions must be positive")
    elif config.pupil_shape == "circle":
        if config.pupil_diameter_mm <= 0:
            raise ValueError("circular pupil diameter must be positive")
    else:
        raise ValueError(f"unsupported pupil shape: {config.pupil_shape}")


def mode_overlap_for_offset(
    config: AngularModeArrayConfig,
    offset_x: np.ndarray | float | int,
    offset_y: np.ndarray | float | int,
) -> np.ndarray:
    """Return ``<psi_i, psi_k>`` for an integer/fractional LED-index offset.

    The paraxial mapping is ``delta_theta = offset * LED_pitch / focal_length``.
    For a uniform rectangular pupil the overlap is a product of sinc functions;
    for a uniform circular pupil it is a jinc function.
    """

    _validate_config(config)
    ox = np.asarray(offset_x, dtype=np.float64)
    oy = np.asarray(offset_y, dtype=np.float64)
    delta_theta_x = ox * config.led_pitch_x_um / config.focal_length_um
    delta_theta_y = oy * config.led_pitch_y_um / config.focal_length_um
    wavelength_um = config.wavelength_um

    if config.pupil_shape == "rectangle":
        width_um = config.pupil_width_mm * 1000.0
        height_um = config.pupil_height_mm * 1000.0
        return (
            np.sinc(width_um * delta_theta_x / wavelength_um)
            * np.sinc(height_um * delta_theta_y / wavelength_um)
        )

    radius_um = config.pupil_diameter_mm * 500.0
    radial_angle = np.hypot(delta_theta_x, delta_theta_y)
    argument = 2.0 * np.pi * radius_um * radial_angle / wavelength_um
    result = np.ones(np.broadcast_shapes(ox.shape, oy.shape), dtype=np.float64)
    nonzero = argument != 0.0
    np.divide(2.0 * j1(argument), argument, out=result, where=nonzero)
    return result


def summarize_regular_array(config: AngularModeArrayConfig) -> ModeOverlapSummary:
    """Compute overlap and participation-ratio rank without forming full Gram.

    Translation invariance gives

    ``Tr(G²) = Σ_dx,dy count(dx,dy) |g(dx,dy)|²``.

    The work and temporary memory therefore scale as O(N_led), rather than
    O(N_led²).
    """

    _validate_config(config)
    offsets_x = np.arange(-(config.layout_x - 1), config.layout_x, dtype=np.int64)
    offsets_y = np.arange(-(config.layout_y - 1), config.layout_y, dtype=np.int64)
    ox, oy = np.meshgrid(offsets_x, offsets_y, indexing="xy")
    magnitude = np.abs(mode_overlap_for_offset(config, ox, oy))

    center_y = config.layout_y - 1
    center_x = config.layout_x - 1
    off_diagonal = magnitude.copy()
    off_diagonal[center_y, center_x] = -np.inf
    flat_index = int(np.argmax(off_diagonal))
    max_y, max_x = np.unravel_index(flat_index, off_diagonal.shape)
    max_overlap = float(magnitude[max_y, max_x])

    pair_count_x = config.layout_x - np.abs(offsets_x)
    pair_count_y = config.layout_y - np.abs(offsets_y)
    pair_count = pair_count_y[:, None] * pair_count_x[None, :]
    trace_g2 = float(np.sum(pair_count * magnitude**2))
    effective_rank = float(config.n_led**2 / trace_g2)

    adjacent_x = (
        float(abs(mode_overlap_for_offset(config, 1, 0)))
        if config.layout_x > 1 else 0.0
    )
    adjacent_y = (
        float(abs(mode_overlap_for_offset(config, 0, 1)))
        if config.layout_y > 1 else 0.0
    )

    step_x = config.led_pitch_x_um / config.focal_length_um
    step_y = config.led_pitch_y_um / config.focal_length_um
    return ModeOverlapSummary(
        n_led=config.n_led,
        adjacent_x_overlap=adjacent_x,
        adjacent_y_overlap=adjacent_y,
        max_off_diagonal_overlap=max_overlap,
        max_overlap_offset_x=int(offsets_x[max_x]),
        max_overlap_offset_y=int(offsets_y[max_y]),
        worst_pair_max_total_variation=float(output_total_variation_bound(max_overlap)),
        effective_rank=effective_rank,
        effective_rank_fraction=effective_rank / config.n_led,
        angular_step_x_rad=step_x,
        angular_step_y_rad=step_y,
        edge_angle_x_rad=0.5 * (config.layout_x - 1) * step_x,
        edge_angle_y_rad=0.5 * (config.layout_y - 1) * step_y,
    )

