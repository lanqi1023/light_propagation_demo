"""General Gram-matrix and output distinguishability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeOverlapSummary:
    """Summary for a regular position-to-angle input-mode array."""

    n_led: int
    adjacent_x_overlap: float
    adjacent_y_overlap: float
    max_off_diagonal_overlap: float
    max_overlap_offset_x: int
    max_overlap_offset_y: int
    worst_pair_max_total_variation: float
    effective_rank: float
    effective_rank_fraction: float
    angular_step_x_rad: float
    angular_step_y_rad: float
    edge_angle_x_rad: float
    edge_angle_y_rad: float


def output_total_variation_bound(overlap: np.ndarray | float) -> np.ndarray:
    """Maximum output-column TV distance for pure modes of given overlap."""

    g = np.clip(np.abs(np.asarray(overlap, dtype=np.float64)), 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - g**2))


def target_pair_feasibility(
    overlap: float,
    column_a: np.ndarray,
    column_b: np.ndarray,
    *,
    atol: float = 1e-12,
) -> dict[str, float | bool]:
    """Check necessary distinguishability constraints for two target columns."""

    a = np.asarray(column_a, dtype=np.float64)
    b = np.asarray(column_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("target columns must have the same shape")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("target intensity columns must be non-negative")
    sum_a = float(np.sum(a))
    sum_b = float(np.sum(b))
    if sum_a <= 0 or sum_b <= 0:
        raise ValueError("target columns must carry positive power")
    a = a / sum_a
    b = b / sum_b

    g = float(np.clip(abs(overlap), 0.0, 1.0))
    total_variation = 0.5 * float(np.sum(np.abs(a - b)))
    max_total_variation = float(output_total_variation_bound(g))
    bhattacharyya = float(np.sum(np.sqrt(a * b)))
    return {
        "overlap": g,
        "total_variation": total_variation,
        "max_total_variation": max_total_variation,
        "bhattacharyya": bhattacharyya,
        "min_bhattacharyya": g,
        "feasible": bool(
            total_variation <= max_total_variation + atol
            and bhattacharyya + atol >= g
        ),
    }


def gram_matrix_from_modes(modes: np.ndarray, *, pixel_area: float = 1.0) -> np.ndarray:
    """Construct a normalized Gram matrix for a manageable mode subset.

    Parameters
    ----------
    modes:
        Complex array shaped ``(n_modes, ...)``. The trailing dimensions are any
        sampled spatial/polarization dimensions.
    pixel_area:
        Quadrature weight for a uniform spatial grid. It cancels after
        normalization but is retained for clarity and future nonuniform weights.
    """

    fields = np.asarray(modes, dtype=np.complex128)
    if fields.ndim < 2:
        raise ValueError("modes must have shape (n_modes, ...)")
    flat = fields.reshape(fields.shape[0], -1)
    norms = np.sqrt(np.sum(np.abs(flat)**2, axis=1) * pixel_area)
    if np.any(norms == 0):
        raise ValueError("all modes must carry non-zero power")
    normalized = flat * np.sqrt(pixel_area) / norms[:, None]
    return normalized @ normalized.conj().T


def gram_diagnostics(gram: np.ndarray, *, eigenvalue_tol: float = 1e-10) -> dict[str, float | int]:
    """Return rank, condition and coherence diagnostics for a Gram matrix."""

    matrix = np.asarray(gram, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be square")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian).real
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(np.sum(eigenvalues))
    trace_square = float(np.sum(eigenvalues**2))
    effective_rank = trace**2 / trace_square if trace_square > 0 else 0.0
    threshold = eigenvalue_tol * max(float(eigenvalues[-1]), 1.0)
    positive = eigenvalues[eigenvalues > threshold]
    numerical_rank = int(positive.size)
    condition_number = (
        float(positive[-1] / positive[0])
        if positive.size else float("inf")
    )
    off_diagonal = np.abs(matrix).copy()
    np.fill_diagonal(off_diagonal, 0.0)
    return {
        "n_modes": matrix.shape[0],
        "numerical_rank": numerical_rank,
        "effective_rank": float(effective_rank),
        "effective_rank_fraction": float(effective_rank / matrix.shape[0]),
        "max_off_diagonal_overlap": float(np.max(off_diagonal, initial=0.0)),
        "condition_number": condition_number,
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
    }


def helstrom_intensity_columns(overlap: float) -> tuple[np.ndarray, np.ndarray]:
    """Construct a lossless two-output measurement that saturates the TV bound."""

    g = float(abs(overlap))
    if not 0.0 <= g <= 1.0:
        raise ValueError("overlap must be between 0 and 1")
    psi_0 = np.array([1.0, 0.0], dtype=np.complex128)
    psi_1 = np.array([g, np.sqrt(max(0.0, 1.0 - g**2))], dtype=np.complex128)
    rho_0 = np.outer(psi_0, psi_0.conj())
    rho_1 = np.outer(psi_1, psi_1.conj())
    _, eigenvectors = np.linalg.eigh(rho_0 - rho_1)
    unitary = eigenvectors.conj().T
    return (
        (np.abs(unitary @ psi_0)**2).real,
        (np.abs(unitary @ psi_1)**2).real,
    )

