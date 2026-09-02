"""End-to-end incoherent multilayer-metasurface models for PyTorch."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from .torch_grid import CartesianGrid
from .torch_optics import (
    AngularSpectrumPropagator,
    AreaDetector,
    CollimatingMetalens,
    PhaseSurface,
    validate_propagation_distances,
)
from .torch_source import LambertianLEDArray


class MultilayerMetasurface(nn.Module):
    """A fixed collimating metalens followed by trainable phase surfaces."""

    def __init__(
        self,
        grid: CartesianGrid,
        wavelength_m: float,
        focal_length_m: float,
        *,
        propagation_distances_m: Sequence[float],
        n_trainable_layers: int,
        aperture: torch.Tensor | None = None,
        trainable_collimator_residual: bool = False,
        include_evanescent: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if n_trainable_layers < 0:
            raise ValueError("n_trainable_layers cannot be negative")
        distances = validate_propagation_distances(
            propagation_distances_m,
            n_trainable_layers,
        )
        self.grid = grid
        self.wavelength_m = float(wavelength_m)
        self.collimator = CollimatingMetalens(
            grid,
            wavelength_m,
            focal_length_m,
            aperture=aperture,
            trainable_residual=trainable_collimator_residual,
            dtype=dtype,
        )
        self.trainable_surfaces = nn.ModuleList(
            [
                PhaseSurface(
                    grid,
                    trainable=True,
                    aperture=aperture,
                    dtype=dtype,
                )
                for _ in range(n_trainable_layers)
            ]
        )
        self.propagators = nn.ModuleList(
            [
                AngularSpectrumPropagator(
                    grid,
                    wavelength_m,
                    distance,
                    include_evanescent=include_evanescent,
                    dtype=dtype,
                )
                for distance in distances
            ]
        )

    def forward(self, field_at_collimator: torch.Tensor) -> torch.Tensor:
        field = self.collimator(field_at_collimator)
        for index, surface in enumerate(self.trainable_surfaces):
            field = self.propagators[index](field)
            field = surface(field)
        return self.propagators[-1](field)


class IncoherentMetasurfaceModel(nn.Module):
    """Map non-negative LED powers to detector powers through the wave system.

    ``method="exact"`` propagates every mutually incoherent source mode and
    sums intensities.  ``method="stochastic"`` propagates random coherent
    superpositions and averages their detected intensities.  Both estimate the
    same mutual-intensity model and both are differentiable with respect to all
    trainable phase surfaces.
    """

    def __init__(
        self,
        source: LambertianLEDArray,
        optical_system: MultilayerMetasurface,
        detector: AreaDetector,
    ) -> None:
        super().__init__()
        if detector.spatial_shape != optical_system.grid.shape:
            raise ValueError("detector and optical system must share a grid")
        if detector.masks.dtype != optical_system.collimator.lens_phase.dtype:
            raise ValueError("detector masks and optical system must share a real dtype")
        if not math.isclose(
            source.config.source_to_metasurface_m,
            optical_system.collimator.focal_length_m,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            raise ValueError(
                "a collimating configuration requires the LED plane distance "
                "to equal the first metalens focal length"
            )
        self.source = source
        self.optical_system = optical_system
        self.detector = detector

    @property
    def n_leds(self) -> int:
        return self.source.config.n_leds

    @property
    def n_detectors(self) -> int:
        return self.detector.n_detectors

    def _as_batched_powers(
        self,
        powers: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        squeeze = powers.ndim == 1
        if squeeze:
            powers = powers.unsqueeze(0)
        if powers.ndim != 2 or powers.shape[1] != self.n_leds:
            raise ValueError(f"powers must have shape (batch, {self.n_leds})")
        if not powers.is_floating_point():
            raise TypeError("powers must be floating point")
        if torch.any(powers < 0):
            raise ValueError("powers must be non-negative")
        reference = self.optical_system.collimator.lens_phase
        if powers.device != reference.device or powers.dtype != reference.dtype:
            raise ValueError(
                "powers must use the same device and real dtype as the model; "
                "move the model and create/cast powers consistently"
            )
        return powers, squeeze

    def forward_exact(
        self,
        powers: torch.Tensor,
        *,
        mode_chunk_size: int = 64,
    ) -> torch.Tensor:
        """Evaluate the deterministic incoherent-mode intensity sum."""

        powers, squeeze = self._as_batched_powers(powers)
        output = torch.zeros(
            (powers.shape[0], self.n_detectors),
            device=powers.device,
            dtype=powers.dtype,
        )
        for modes, led_indices in self.source.iter_unit_modes(
            self.optical_system.grid,
            self.optical_system.wavelength_m,
            chunk_size=mode_chunk_size,
            device=powers.device,
            dtype=powers.dtype,
        ):
            detected_modes = self.detector(self.optical_system(modes))
            output = output + torch.einsum(
                "bm,md->bd",
                powers[:, led_indices],
                detected_modes,
            )
        return output.squeeze(0) if squeeze else output

    def forward_stochastic(
        self,
        powers: torch.Tensor,
        *,
        n_realizations: int = 8,
        realization_chunk_size: int | None = None,
        mode_chunk_size: int = 64,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Evaluate an unbiased random-phase estimate of detector powers."""

        powers, squeeze = self._as_batched_powers(powers)
        if n_realizations <= 0:
            raise ValueError("n_realizations must be positive")
        if realization_chunk_size is None:
            realization_chunk_size = n_realizations
        if realization_chunk_size <= 0:
            raise ValueError("realization_chunk_size must be positive")

        batch = powers.shape[0]
        output_sum = torch.zeros(
            (batch, self.n_detectors),
            device=powers.device,
            dtype=powers.dtype,
        )
        completed = 0
        while completed < n_realizations:
            count = min(realization_chunk_size, n_realizations - completed)
            fields = self.source.stochastic_fields(
                powers,
                self.optical_system.grid,
                self.optical_system.wavelength_m,
                n_realizations=count,
                chunk_size=mode_chunk_size,
                generator=generator,
            )
            propagated = self.optical_system(
                fields.reshape(batch * count, *self.optical_system.grid.shape)
            )
            detected = self.detector(propagated).reshape(
                batch,
                count,
                self.n_detectors,
            )
            output_sum = output_sum + detected.sum(dim=1)
            completed += count
        output = output_sum / n_realizations
        return output.squeeze(0) if squeeze else output

    def forward(
        self,
        powers: torch.Tensor,
        *,
        method: Literal["exact", "stochastic"] = "stochastic",
        n_realizations: int = 8,
        realization_chunk_size: int | None = None,
        mode_chunk_size: int = 64,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if method == "exact":
            return self.forward_exact(powers, mode_chunk_size=mode_chunk_size)
        if method == "stochastic":
            return self.forward_stochastic(
                powers,
                n_realizations=n_realizations,
                realization_chunk_size=realization_chunk_size,
                mode_chunk_size=mode_chunk_size,
                generator=generator,
            )
        raise ValueError(f"unsupported incoherent propagation method: {method}")

    def intensity_matrix(self, *, mode_chunk_size: int = 64) -> torch.Tensor:
        """Construct ``A`` with shape ``(n_detectors, n_leds)`` for diagnostics.

        This is exact but intended only for small systems or offline chunked
        validation.  Training a 49,152-input system should call ``forward`` on
        input vectors instead of materializing the full matrix.
        """

        reference = self.optical_system.collimator.lens_phase
        device = reference.device
        dtype = reference.dtype
        matrix = torch.zeros(
            (self.n_detectors, self.n_leds),
            device=device,
            dtype=dtype,
        )
        for modes, led_indices in self.source.iter_unit_modes(
            self.optical_system.grid,
            self.optical_system.wavelength_m,
            chunk_size=mode_chunk_size,
            device=device,
            dtype=dtype,
        ):
            detected_modes = self.detector(self.optical_system(modes))
            matrix = matrix.index_add(1, led_indices, detected_modes.transpose(0, 1))
        return matrix
