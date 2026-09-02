"""Uniform Cartesian sampling grids used by the PyTorch wave model."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CartesianGrid:
    """A centered, uniformly sampled transverse plane in SI units.

    The sampled complex field uses a power-normalized discrete convention:
    ``abs(field[..., y, x]) ** 2`` is the power carried by that sample.  The
    pixel-area factor is therefore introduced when continuous source fields are
    sampled, rather than inside every propagation operation.
    """

    height: int
    width: int
    spacing_y_m: float
    spacing_x_m: float

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.spacing_y_m <= 0 or self.spacing_x_m <= 0:
            raise ValueError("grid spacings must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def pixel_area_m2(self) -> float:
        return self.spacing_y_m * self.spacing_x_m

    def coordinates(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return centered ``(y, x)`` coordinate arrays shaped ``(H, W)``."""

        y = (
            torch.arange(self.height, device=device, dtype=dtype)
            - 0.5 * (self.height - 1)
        ) * self.spacing_y_m
        x = (
            torch.arange(self.width, device=device, dtype=dtype)
            - 0.5 * (self.width - 1)
        ) * self.spacing_x_m
        return torch.meshgrid(y, x, indexing="ij")

    def spatial_frequencies(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FFT-ordered spatial frequencies ``(fy, fx)`` in cycles/m."""

        fy = torch.fft.fftfreq(
            self.height,
            d=self.spacing_y_m,
            device=device,
            dtype=dtype,
        )
        fx = torch.fft.fftfreq(
            self.width,
            d=self.spacing_x_m,
            device=device,
            dtype=dtype,
        )
        return torch.meshgrid(fy, fx, indexing="ij")

