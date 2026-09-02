"""MicroLED input-mode diagnostics for incoherent metasurface processors."""

from .diagnostics import (
    ModeOverlapSummary,
    gram_diagnostics,
    gram_matrix_from_modes,
    helstrom_intensity_columns,
    output_total_variation_bound,
    target_pair_feasibility,
)
from .model import AngularModeArrayConfig, mode_overlap_for_offset, summarize_regular_array

__all__ = [
    "AngularModeArrayConfig",
    "ModeOverlapSummary",
    "gram_diagnostics",
    "gram_matrix_from_modes",
    "helstrom_intensity_columns",
    "mode_overlap_for_offset",
    "output_total_variation_bound",
    "summarize_regular_array",
    "target_pair_feasibility",
]

