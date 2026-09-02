"""Lambertian MicroLED-array modes for incoherent PyTorch simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

from .torch_grid import CartesianGrid


@dataclass(frozen=True)
class LambertianLEDArrayConfig:
    """Geometry of a regular planar MicroLED array in SI units.

    Each LED is divided into ``emitter_samples_y * emitter_samples_x`` mutually
    incoherent point emitters.  The samples share that LED's total input power
    equally.  Set both sample counts to one for the point-LED approximation.
    """

    layout_y: int
    layout_x: int
    pitch_y_m: float
    pitch_x_m: float
    source_to_metasurface_m: float
    emitter_height_m: float = 0.0
    emitter_width_m: float = 0.0
    emitter_samples_y: int = 1
    emitter_samples_x: int = 1

    def __post_init__(self) -> None:
        if self.layout_y <= 0 or self.layout_x <= 0:
            raise ValueError("LED layout dimensions must be positive")
        if self.pitch_y_m <= 0 or self.pitch_x_m <= 0:
            raise ValueError("LED pitches must be positive")
        if self.source_to_metasurface_m <= 0:
            raise ValueError("source-to-metasurface distance must be positive")
        if self.emitter_height_m < 0 or self.emitter_width_m < 0:
            raise ValueError("emitter dimensions cannot be negative")
        if self.emitter_samples_y <= 0 or self.emitter_samples_x <= 0:
            raise ValueError("emitter sample counts must be positive")

    @property
    def n_leds(self) -> int:
        return self.layout_y * self.layout_x

    @property
    def samples_per_led(self) -> int:
        return self.emitter_samples_y * self.emitter_samples_x

    @property
    def n_modes(self) -> int:
        return self.n_leds * self.samples_per_led


def circular_aperture(
    grid: CartesianGrid,
    diameter_m: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a centered circular amplitude aperture."""

    if diameter_m <= 0:
        raise ValueError("aperture diameter must be positive")
    y, x = grid.coordinates(device=device, dtype=dtype)
    return (x.square() + y.square() <= (0.5 * diameter_m) ** 2).to(dtype)


def rectangular_aperture(
    grid: CartesianGrid,
    height_m: float,
    width_m: float,
    *,
    center_y_m: float = 0.0,
    center_x_m: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a rectangular amplitude aperture on the sampled grid."""

    if height_m <= 0 or width_m <= 0:
        raise ValueError("aperture height and width must be positive")
    y, x = grid.coordinates(device=device, dtype=dtype)
    inside_y = torch.abs(y - center_y_m) <= 0.5 * height_m
    inside_x = torch.abs(x - center_x_m) <= 0.5 * width_m
    return (inside_y & inside_x).to(dtype)


class LambertianLEDArray(nn.Module):
    r"""A spatially incoherent array of finite Lambertian emitters.

    For a unit-power Lambertian point emitter and a parallel receiving plane,
    the irradiance is

    .. math::

        E(x,y)=\frac{1}{\pi}\frac{\cos^2\theta}{R^2}.

    The returned discrete mode therefore has magnitude
    ``sqrt(pixel_area / pi) * cos(theta) / R``.  Its squared norm is the
    fraction of emitted power intercepted by the sampled aperture.  Modes are
    deliberately *not* normalized to one, so collection loss remains visible.
    """

    def __init__(
        self,
        config: LambertianLEDArrayConfig,
        *,
        aperture: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        positions, led_indices = self._make_mode_coordinates(config)
        self.register_buffer("mode_positions_m", positions, persistent=True)
        self.register_buffer("mode_led_indices", led_indices, persistent=True)
        if aperture is None:
            self.aperture = None
        else:
            if aperture.ndim != 2:
                raise ValueError("aperture must have shape (H, W)")
            self.register_buffer("aperture", aperture.detach().clone(), persistent=True)

    @staticmethod
    def _sample_offsets(size_m: float, count: int) -> torch.Tensor:
        if count == 1 or size_m == 0.0:
            return torch.zeros(count, dtype=torch.float64)
        # Midpoint quadrature avoids placing samples exactly on LED boundaries.
        return (
            (torch.arange(count, dtype=torch.float64) + 0.5) / count - 0.5
        ) * size_m

    @classmethod
    def _make_mode_coordinates(
        cls,
        config: LambertianLEDArrayConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        led_y = (
            torch.arange(config.layout_y, dtype=torch.float64)
            - 0.5 * (config.layout_y - 1)
        ) * config.pitch_y_m
        led_x = (
            torch.arange(config.layout_x, dtype=torch.float64)
            - 0.5 * (config.layout_x - 1)
        ) * config.pitch_x_m
        centers_y, centers_x = torch.meshgrid(led_y, led_x, indexing="ij")
        centers = torch.stack((centers_y.reshape(-1), centers_x.reshape(-1)), dim=-1)

        offset_y = cls._sample_offsets(
            config.emitter_height_m,
            config.emitter_samples_y,
        )
        offset_x = cls._sample_offsets(
            config.emitter_width_m,
            config.emitter_samples_x,
        )
        sample_y, sample_x = torch.meshgrid(offset_y, offset_x, indexing="ij")
        offsets = torch.stack((sample_y.reshape(-1), sample_x.reshape(-1)), dim=-1)

        positions = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        led_indices = torch.arange(config.n_leds, dtype=torch.long).repeat_interleave(
            config.samples_per_led
        )
        return positions, led_indices

    def _aperture_for(
        self,
        grid: CartesianGrid,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.aperture is None:
            return torch.ones(grid.shape, device=device, dtype=dtype)
        if tuple(self.aperture.shape) != grid.shape:
            raise ValueError(
                f"source aperture has shape {tuple(self.aperture.shape)}, "
                f"expected {grid.shape}"
            )
        return self.aperture.to(device=device, dtype=dtype)

    def unit_modes(
        self,
        grid: CartesianGrid,
        wavelength_m: float,
        start: int = 0,
        stop: int | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a chunk of sampled unit-input modes and their LED indices.

        The modes have shape ``(n_chunk, H, W)`` and a complex dtype derived
        from ``dtype``.  The finite-emitter quadrature weight is already folded
        into their amplitude.
        """

        if wavelength_m <= 0:
            raise ValueError("wavelength must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")
        stop = self.config.n_modes if stop is None else stop
        if not 0 <= start <= stop <= self.config.n_modes:
            raise ValueError("invalid mode slice")

        target_device = torch.device(device) if device is not None else self.mode_positions_m.device
        positions = self.mode_positions_m[start:stop].to(device=target_device, dtype=dtype)
        led_indices = self.mode_led_indices[start:stop].to(device=target_device)
        y, x = grid.coordinates(device=target_device, dtype=dtype)
        source_y = positions[:, 0, None, None]
        source_x = positions[:, 1, None, None]
        dz = torch.as_tensor(
            self.config.source_to_metasurface_m,
            device=target_device,
            dtype=dtype,
        )
        radius = torch.sqrt(
            (y[None] - source_y).square()
            + (x[None] - source_x).square()
            + dz.square()
        )
        cosine = dz / radius

        sample_weight = 1.0 / self.config.samples_per_led
        amplitude = (
            math.sqrt(grid.pixel_area_m2 * sample_weight / math.pi)
            * cosine
            / radius
        )
        amplitude = amplitude * self._aperture_for(
            grid,
            device=target_device,
            dtype=dtype,
        )[None]
        phase = (2.0 * math.pi / wavelength_m) * radius
        return torch.polar(amplitude, phase), led_indices

    def iter_unit_modes(
        self,
        grid: CartesianGrid,
        wavelength_m: float,
        *,
        chunk_size: int = 64,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield unit modes without storing the full LED-mode tensor."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for start in range(0, self.config.n_modes, chunk_size):
            yield self.unit_modes(
                grid,
                wavelength_m,
                start,
                min(start + chunk_size, self.config.n_modes),
                device=device,
                dtype=dtype,
            )

    def stochastic_fields(
        self,
        powers: torch.Tensor,
        grid: CartesianGrid,
        wavelength_m: float,
        *,
        n_realizations: int,
        chunk_size: int = 64,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample random coherent realizations of the incoherent LED mixture.

        ``powers`` is shaped ``(batch, n_leds)`` or ``(n_leds,)``.  Independent
        random phases are assigned to all LED sub-emitter modes.  The result is
        shaped ``(batch, n_realizations, H, W)``.
        """

        if n_realizations <= 0:
            raise ValueError("n_realizations must be positive")
        if not powers.is_floating_point():
            raise TypeError("LED powers must be a floating-point tensor")
        squeeze_batch = powers.ndim == 1
        if squeeze_batch:
            powers = powers.unsqueeze(0)
        if powers.ndim != 2 or powers.shape[1] != self.config.n_leds:
            raise ValueError(
                f"powers must have shape (batch, {self.config.n_leds})"
            )
        if torch.any(powers < 0):
            raise ValueError("LED powers must be non-negative")

        batch = powers.shape[0]
        fields = torch.zeros(
            (batch, n_realizations, *grid.shape),
            device=powers.device,
            dtype=(torch.complex64 if powers.dtype == torch.float32 else torch.complex128),
        )
        for modes, led_indices in self.iter_unit_modes(
            grid,
            wavelength_m,
            chunk_size=chunk_size,
            device=powers.device,
            dtype=powers.dtype,
        ):
            phase = 2.0 * math.pi * torch.rand(
                (batch, n_realizations, modes.shape[0]),
                device=powers.device,
                dtype=powers.dtype,
                generator=generator,
            )
            magnitude = torch.sqrt(powers[:, led_indices])[:, None, :]
            coefficients = torch.polar(magnitude.expand_as(phase), phase)
            fields = fields + torch.einsum("bkm,mhw->bkhw", coefficients, modes)
        return fields.squeeze(0) if squeeze_batch else fields

    @torch.no_grad()
    def collection_efficiencies(
        self,
        grid: CartesianGrid,
        wavelength_m: float,
        *,
        chunk_size: int = 64,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the first-surface collected fraction for every LED."""

        target_device = torch.device(device) if device is not None else self.mode_positions_m.device
        result = torch.zeros(self.config.n_leds, device=target_device, dtype=dtype)
        for modes, led_indices in self.iter_unit_modes(
            grid,
            wavelength_m,
            chunk_size=chunk_size,
            device=target_device,
            dtype=dtype,
        ):
            mode_power = modes.abs().square().sum(dim=(-2, -1))
            result.index_add_(0, led_indices, mode_power)
        return result
