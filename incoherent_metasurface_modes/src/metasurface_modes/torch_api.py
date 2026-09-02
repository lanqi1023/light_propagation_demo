"""Public PyTorch API, kept optional from the NumPy-only diagnostics package."""

from .torch_grid import CartesianGrid
from .torch_optics import (
    AngularSpectrumPropagator,
    AreaDetector,
    CollimatingMetalens,
    PhaseSurface,
    tiled_detector_masks,
)
from .torch_source import (
    LambertianLEDArray,
    LambertianLEDArrayConfig,
    circular_aperture,
    rectangular_aperture,
)
from .torch_system import IncoherentMetasurfaceModel, MultilayerMetasurface

__all__ = [
    "AngularSpectrumPropagator",
    "AreaDetector",
    "CartesianGrid",
    "CollimatingMetalens",
    "IncoherentMetasurfaceModel",
    "LambertianLEDArray",
    "LambertianLEDArrayConfig",
    "MultilayerMetasurface",
    "PhaseSurface",
    "circular_aperture",
    "rectangular_aperture",
    "tiled_detector_masks",
]
