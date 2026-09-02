"""A compact, fully verifiable incoherent-metasurface demonstration.

The demo performs three checks:

1. exact incoherent propagation is a non-negative matrix multiplication;
2. the random-phase estimate converges to the exact detector powers;
3. gradients can train a phase surface toward a realizable teacher system.
"""

from __future__ import annotations

import torch

from metasurface_modes.torch_api import (
    AreaDetector,
    CartesianGrid,
    IncoherentMetasurfaceModel,
    LambertianLEDArray,
    LambertianLEDArrayConfig,
    MultilayerMetasurface,
    rectangular_aperture,
    tiled_detector_masks,
)


def build_model(device: torch.device) -> IncoherentMetasurfaceModel:
    wavelength_m = 532e-9
    focal_length_m = 20e-3
    grid = CartesianGrid(192, 216, 6e-6, 6e-6)
    aperture = rectangular_aperture(
        grid,
        height_m=1.0e-3,
        width_m=1.2e-3,
        device=device,
    )
    source = LambertianLEDArray(
        LambertianLEDArrayConfig(
            layout_y=2,
            layout_x=2,
            pitch_y_m=35e-6,
            pitch_x_m=35e-6,
            source_to_metasurface_m=focal_length_m,
            emitter_height_m=4e-6,
            emitter_width_m=4e-6,
            emitter_samples_y=1,
            emitter_samples_x=1,
        ),
        aperture=aperture,
    )
    optical = MultilayerMetasurface(
        grid,
        wavelength_m,
        focal_length_m,
        n_trainable_layers=1,
        propagation_distances_m=(5e-3, 5e-3),
        aperture=aperture,
    )
    detector = AreaDetector(tiled_detector_masks(grid, 2, 2, device=device))
    return IncoherentMetasurfaceModel(source, optical, detector).to(device)


def set_teacher_phase(model: IncoherentMetasurfaceModel) -> None:
    """Set a smooth, exactly representable target phase pattern."""

    grid = model.optical_system.grid
    phase = model.optical_system.trainable_surfaces[0].phase
    y, x = grid.coordinates(device=phase.device, dtype=phase.dtype)
    width_x = grid.width * grid.spacing_x_m
    width_y = grid.height * grid.spacing_y_m
    pattern = (
        0.8 * torch.sin(2.0 * torch.pi * x / width_x)
        + 0.6 * torch.cos(2.0 * torch.pi * y / width_y)
        + 0.3 * torch.sin(2.0 * torch.pi * (x + y) / width_x)
    )
    with torch.no_grad():
        phase.copy_(pattern)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    ).clamp_min(1e-16)


def main() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")

    # A teacher with the same architecture makes the target realizable by
    # construction.  It is only used to produce target detector powers.
    teacher = build_model(device)
    set_teacher_phase(teacher)
    teacher.requires_grad_(False)
    with torch.no_grad():
        target_matrix = teacher.intensity_matrix(mode_chunk_size=4)

    student = build_model(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.08)
    for step in range(151):
        predicted_matrix = student.intensity_matrix(mode_chunk_size=4)
        loss = (
            (predicted_matrix - target_matrix).square().mean()
            / target_matrix.square().mean().clamp_min(1e-16)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 30 == 0:
            print(f"train step={step:03d} normalized_mse={loss.item():.6e}")

    with torch.no_grad():
        fitted_matrix = student.intensity_matrix(mode_chunk_size=4)
        matrix_error = relative_error(fitted_matrix, target_matrix)

        powers = torch.tensor(
            [[1.0, 0.3, 0.6, 0.1], [0.2, 0.8, 0.4, 1.0]],
            device=device,
        )
        exact = student(powers, method="exact", mode_chunk_size=4)
        matrix_product = powers @ fitted_matrix.transpose(0, 1)
        linearity_error = relative_error(exact, matrix_product)

        generator = torch.Generator(device=device).manual_seed(9)
        stochastic = student(
            powers,
        method="stochastic",
        n_realizations=4096,
        realization_chunk_size=64,
        mode_chunk_size=4,
            generator=generator,
        )
        monte_carlo_error = relative_error(stochastic, exact)
        collection = student.source.collection_efficiencies(
            student.optical_system.grid,
            student.optical_system.wavelength_m,
            device=device,
        )

    print(f"matrix_fit_relative_error={matrix_error.item():.6e}")
    print(f"exact_vs_Ax_relative_error={linearity_error.item():.6e}")
    print(f"stochastic_vs_exact_relative_error={monte_carlo_error.item():.6e}")
    print(
        "first_surface_collection_efficiency="
        f"[{collection.min().item():.6f}, {collection.max().item():.6f}]"
    )
    print("example_exact_detector_powers=")
    print(exact.cpu())


if __name__ == "__main__":
    main()
