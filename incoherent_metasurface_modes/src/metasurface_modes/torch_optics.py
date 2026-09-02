"""Differentiable scalar-wave components for multilayer metasurfaces."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .torch_grid import CartesianGrid


class AngularSpectrumPropagator(nn.Module):
    """FFT angular-spectrum propagation on a fixed Cartesian grid.

    Orthogonal FFT normalization makes the propagating part power preserving.
    Spatial frequencies outside the free-space propagation disk are discarded
    by default.  Padding is still the caller's responsibility when wrap-around
    from the FFT's periodic boundary condition would be significant.
    """

    def __init__(
        self,
        grid: CartesianGrid,
        wavelength_m: float,
        distance_m: float,
        *,
        include_evanescent: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if wavelength_m <= 0:
            raise ValueError("wavelength must be positive")
        if distance_m < 0:
            raise ValueError("distance must be non-negative")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")
        self.grid = grid
        self.wavelength_m = float(wavelength_m)
        self.distance_m = float(distance_m)

        fy, fx = grid.spatial_frequencies(dtype=dtype)
        k = 2.0 * math.pi / wavelength_m
        ky = 2.0 * math.pi * fy
        kx = 2.0 * math.pi * fx
        transverse_square = kx.square() + ky.square()
        propagating = transverse_square <= k * k
        kz = torch.sqrt(torch.clamp(k * k - transverse_square, min=0.0))
        transfer = torch.polar(torch.ones_like(kz), distance_m * kz)
        if include_evanescent:
            decay = torch.exp(
                -distance_m * torch.sqrt(
                    torch.clamp(transverse_square - k * k, min=0.0)
                )
            )
            transfer = torch.where(propagating, transfer, decay.to(transfer.dtype))
        else:
            transfer = transfer * propagating
        self.register_buffer("transfer", transfer, persistent=True)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.shape[-2:] != self.grid.shape:
            raise ValueError(
                f"field has spatial shape {tuple(field.shape[-2:])}, "
                f"expected {self.grid.shape}"
            )
        spectrum = torch.fft.fft2(field, dim=(-2, -1), norm="ortho")
        return torch.fft.ifft2(
            spectrum * self.transfer,
            dim=(-2, -1),
            norm="ortho",
        )


class PhaseSurface(nn.Module):
    """A local phase-only surface with an optional fixed amplitude aperture."""

    def __init__(
        self,
        grid: CartesianGrid,
        *,
        initial_phase: torch.Tensor | None = None,
        trainable: bool = True,
        aperture: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.grid = grid
        if initial_phase is None:
            initial_phase = torch.zeros(grid.shape, dtype=dtype)
        if tuple(initial_phase.shape) != grid.shape:
            raise ValueError("initial_phase must match the grid shape")
        phase = initial_phase.detach().clone().to(dtype=dtype)
        if trainable:
            self.phase = nn.Parameter(phase)
        else:
            self.register_buffer("phase", phase, persistent=True)

        if aperture is None:
            aperture = torch.ones(grid.shape, dtype=dtype)
        if tuple(aperture.shape) != grid.shape:
            raise ValueError("aperture must match the grid shape")
        self.register_buffer(
            "aperture",
            aperture.detach().clone().to(dtype=dtype),
            persistent=True,
        )

    def transmission(self) -> torch.Tensor:
        return self.aperture * torch.exp(1j * self.phase)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.shape[-2:] != self.grid.shape:
            raise ValueError("field and phase-surface grid do not match")
        return field * self.transmission()


class CollimatingMetalens(nn.Module):
    r"""A hyperbolic collimating phase with an optional trainable residual.

    For a centered point source at distance ``focal_length_m``, the phase

    .. math::

        -k(\sqrt{f^2+x^2+y^2}-f)

    exactly cancels the sampled spherical-wave phase in this scalar model.
    """

    def __init__(
        self,
        grid: CartesianGrid,
        wavelength_m: float,
        focal_length_m: float,
        *,
        aperture: torch.Tensor | None = None,
        trainable_residual: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if wavelength_m <= 0 or focal_length_m <= 0:
            raise ValueError("wavelength and focal length must be positive")
        self.grid = grid
        self.focal_length_m = float(focal_length_m)
        y, x = grid.coordinates(dtype=dtype)
        optical_path_difference = torch.sqrt(
            focal_length_m**2 + x.square() + y.square()
        ) - focal_length_m
        lens_phase = -(2.0 * math.pi / wavelength_m) * optical_path_difference
        self.register_buffer("lens_phase", lens_phase, persistent=True)

        residual = torch.zeros(grid.shape, dtype=dtype)
        if trainable_residual:
            self.residual_phase = nn.Parameter(residual)
        else:
            self.register_buffer("residual_phase", residual, persistent=True)

        if aperture is None:
            aperture = torch.ones(grid.shape, dtype=dtype)
        if tuple(aperture.shape) != grid.shape:
            raise ValueError("aperture must match the grid shape")
        self.register_buffer(
            "aperture",
            aperture.detach().clone().to(dtype=dtype),
            persistent=True,
        )

    @property
    def total_phase(self) -> torch.Tensor:
        return self.lens_phase + self.residual_phase

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.shape[-2:] != self.grid.shape:
            raise ValueError("field and metalens grid do not match")
        return field * self.aperture * torch.exp(1j * self.total_phase)


class AreaDetector(nn.Module):
    """Square-law detection followed by non-negative spatial integration."""

    def __init__(self, masks: torch.Tensor) -> None:
        super().__init__()
        if masks.ndim != 3:
            raise ValueError("detector masks must have shape (n_detectors, H, W)")
        if not masks.is_floating_point():
            masks = masks.float()
        if torch.any(masks < 0):
            raise ValueError("detector masks must be non-negative")
        self.register_buffer("masks", masks.detach().clone(), persistent=True)

    @property
    def n_detectors(self) -> int:
        return self.masks.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.masks.shape[-2], self.masks.shape[-1]

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.shape[-2:] != self.spatial_shape:
            raise ValueError("field and detector masks do not share a grid")
        intensity = field.abs().square()
        return torch.einsum("...hw,dhw->...d", intensity, self.masks)


def tiled_detector_masks(
    grid: CartesianGrid,
    tiles_y: int,
    tiles_x: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Partition every grid sample into one of ``tiles_y * tiles_x`` bins."""

    if tiles_y <= 0 or tiles_x <= 0:
        raise ValueError("detector tile counts must be positive")
    if tiles_y > grid.height or tiles_x > grid.width:
        raise ValueError("cannot have more detector tiles than grid samples")
    row_bin = torch.div(
        torch.arange(grid.height, device=device) * tiles_y,
        grid.height,
        rounding_mode="floor",
    )
    col_bin = torch.div(
        torch.arange(grid.width, device=device) * tiles_x,
        grid.width,
        rounding_mode="floor",
    )
    bin_index = row_bin[:, None] * tiles_x + col_bin[None, :]
    ids = torch.arange(tiles_y * tiles_x, device=device)[:, None, None]
    return (ids == bin_index[None]).to(dtype=dtype)


def validate_propagation_distances(
    distances_m: Sequence[float],
    n_trainable_layers: int,
) -> tuple[float, ...]:
    """Validate collimator-to-layers-to-detector distances."""

    distances = tuple(float(value) for value in distances_m)
    expected = n_trainable_layers + 1
    if len(distances) != expected:
        raise ValueError(
            f"expected {expected} propagation distances for "
            f"{n_trainable_layers} trainable layers, got {len(distances)}"
        )
    if any(value < 0 for value in distances):
        raise ValueError("propagation distances must be non-negative")
    return distances
