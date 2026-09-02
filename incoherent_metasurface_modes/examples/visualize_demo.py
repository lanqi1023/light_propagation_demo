"""Generate quantitative PNG visualizations from the PyTorch wave model."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Keep Matplotlib's runtime cache in a writable, disposable location.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "incoherent-metasurface-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from simple_demo import build_model, relative_error, set_teacher_phase


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def normalized_intensity(field: torch.Tensor) -> np.ndarray:
    intensity = field.abs().square()
    return as_numpy(intensity / intensity.max().clamp_min(1e-30))


def wrapped_phase(
    field_or_phase: torch.Tensor,
    *,
    is_phase: bool = False,
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    phase = field_or_phase if is_phase else torch.angle(field_or_phase)
    wrapped = as_numpy(torch.atan2(torch.sin(phase), torch.cos(phase)))
    if mask is None and not is_phase:
        intensity = field_or_phase.abs().square()
        mask = intensity > intensity.max().clamp_min(1e-30) * 1e-8
    if mask is not None:
        return np.ma.masked_where(~as_numpy(mask.bool()), wrapped)
    return wrapped


def exact_output_intensity(
    model,
    powers: torch.Tensor,
    *,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Return the exact spatial intensity before detector integration."""

    output = torch.zeros(
        model.optical_system.grid.shape,
        device=powers.device,
        dtype=powers.dtype,
    )
    for modes, led_indices in model.source.iter_unit_modes(
        model.optical_system.grid,
        model.optical_system.wavelength_m,
        chunk_size=chunk_size,
        device=powers.device,
        dtype=powers.dtype,
    ):
        propagated = model.optical_system(modes)
        output = output + torch.einsum(
            "m,mhw->hw",
            powers[led_indices],
            propagated.abs().square(),
        )
    return output


def stochastic_output_intensity(
    model,
    powers: torch.Tensor,
    n_realizations: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=powers.device).manual_seed(seed)
    output_sum = torch.zeros(
        model.optical_system.grid.shape,
        device=powers.device,
        dtype=powers.dtype,
    )
    completed = 0
    while completed < n_realizations:
        count = min(64, n_realizations - completed)
        fields = model.source.stochastic_fields(
            powers,
            model.optical_system.grid,
            model.optical_system.wavelength_m,
            n_realizations=count,
            chunk_size=4,
            generator=generator,
        )
        output_sum = output_sum + model.optical_system(fields).abs().square().sum(dim=0)
        completed += count
    return output_sum / n_realizations


def train_student(device: torch.device):
    teacher = build_model(device)
    set_teacher_phase(teacher)
    teacher.requires_grad_(False)
    with torch.no_grad():
        target_matrix = teacher.intensity_matrix(mode_chunk_size=4)

    student = build_model(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.08)
    losses: list[float] = []
    for _ in range(151):
        fitted = student.intensity_matrix(mode_chunk_size=4)
        loss = (
            (fitted - target_matrix).square().mean()
            / target_matrix.square().mean().clamp_min(1e-16)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        fitted_matrix = student.intensity_matrix(mode_chunk_size=4)
    return teacher, student, target_matrix, fitted_matrix, np.asarray(losses)


def spatial_extent_mm(model) -> tuple[float, float, float, float]:
    grid = model.optical_system.grid
    half_x = 0.5 * (grid.width - 1) * grid.spacing_x_m * 1e3
    half_y = 0.5 * (grid.height - 1) * grid.spacing_y_m * 1e3
    return -half_x, half_x, -half_y, half_y


def add_image(
    figure,
    axis,
    data,
    title: str,
    extent,
    *,
    cmap="viridis",
    vmin=None,
    vmax=None,
    colorbar_label="",
):
    image = axis.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    return image


def plot_optical_fields(student, powers: torch.Tensor) -> Path:
    grid = student.optical_system.grid
    extent = spatial_extent_mm(student)
    with torch.no_grad():
        source_mode, _ = student.source.unit_modes(
            grid,
            student.optical_system.wavelength_m,
            0,
            1,
            device=powers.device,
            dtype=powers.dtype,
        )
        before_lens = source_mode[0]
        after_lens = student.optical_system.collimator(before_lens)
        before_trainable = student.optical_system.propagators[0](after_lens)
        after_trainable = student.optical_system.trainable_surfaces[0](before_trainable)
        final_mode = student.optical_system.propagators[-1](after_trainable)
        incoherent_output = exact_output_intensity(student, powers)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    add_image(
        figure,
        axes[0, 0],
        normalized_intensity(before_lens),
        "LED 0 field at collimating metalens",
        extent,
        colorbar_label="normalized intensity",
    )
    add_image(
        figure,
        axes[0, 1],
        wrapped_phase(before_lens),
        "Spherical-wave phase before metalens",
        extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        colorbar_label="phase (rad)",
    )
    add_image(
        figure,
        axes[0, 2],
        wrapped_phase(after_lens),
        "Phase after collimating metalens",
        extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        colorbar_label="phase (rad)",
    )
    add_image(
        figure,
        axes[1, 0],
        wrapped_phase(
            student.optical_system.trainable_surfaces[0].phase,
            is_phase=True,
            mask=student.optical_system.trainable_surfaces[0].aperture,
        ),
        "Learned trainable-surface phase",
        extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        colorbar_label="phase (rad)",
    )
    add_image(
        figure,
        axes[1, 1],
        normalized_intensity(final_mode),
        "Final coherent intensity for LED 0",
        extent,
        colorbar_label="normalized intensity",
    )
    incoherent_normalized = as_numpy(
        incoherent_output / incoherent_output.max().clamp_min(1e-30)
    )
    add_image(
        figure,
        axes[1, 2],
        incoherent_normalized,
        "Exact incoherent output for 4 LEDs",
        extent,
        colorbar_label="normalized intensity",
    )
    figure.suptitle(
        "Wave propagation through the collimator and trainable metasurface",
        fontsize=15,
    )
    path = OUTPUT_DIR / "optical_fields.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_matrix_training(
    teacher,
    student,
    target_matrix: torch.Tensor,
    fitted_matrix: torch.Tensor,
    losses: np.ndarray,
) -> Path:
    target = as_numpy(target_matrix)
    fitted = as_numpy(fitted_matrix)
    residual = (fitted - target) / max(float(np.max(target)), 1e-30)
    shared_max = max(float(np.max(target)), float(np.max(fitted)))
    phase_target = wrapped_phase(
        teacher.optical_system.trainable_surfaces[0].phase,
        is_phase=True,
        mask=teacher.optical_system.trainable_surfaces[0].aperture,
    )
    phase_fitted = wrapped_phase(
        student.optical_system.trainable_surfaces[0].phase,
        is_phase=True,
        mask=student.optical_system.trainable_surfaces[0].aperture,
    )
    extent = spatial_extent_mm(student)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].semilogy(np.arange(losses.size), losses, color="#1565c0")
    axes[0, 0].set_title("Training convergence")
    axes[0, 0].set_xlabel("optimization step")
    axes[0, 0].set_ylabel("normalized MSE")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    for axis, data, title in (
        (axes[0, 1], target, "Target intensity matrix A"),
        (axes[0, 2], fitted, "Fitted intensity matrix A"),
    ):
        image = axis.imshow(data, cmap="magma", vmin=0.0, vmax=shared_max)
        axis.set_title(title)
        axis.set_xlabel("LED index")
        axis.set_ylabel("detector index")
        axis.set_xticks(range(target.shape[1]))
        axis.set_yticks(range(target.shape[0]))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="power coupling")

    residual_limit = max(float(np.max(np.abs(residual))), 1e-12)
    image = axes[1, 0].imshow(
        residual,
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[1, 0].set_title("Matrix residual / max(target)")
    axes[1, 0].set_xlabel("LED index")
    axes[1, 0].set_ylabel("detector index")
    axes[1, 0].set_xticks(range(target.shape[1]))
    axes[1, 0].set_yticks(range(target.shape[0]))
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    add_image(
        figure,
        axes[1, 1],
        phase_target,
        "Teacher phase",
        extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        colorbar_label="phase (rad)",
    )
    add_image(
        figure,
        axes[1, 2],
        phase_fitted,
        "Learned phase (not uniquely determined)",
        extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        colorbar_label="phase (rad)",
    )
    matrix_error = float(relative_error(fitted_matrix, target_matrix))
    figure.suptitle(
        f"Differentiable training: matrix relative error = {matrix_error:.3e}",
        fontsize=15,
    )
    path = OUTPUT_DIR / "matrix_training.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_stochastic_comparison(student, powers: torch.Tensor) -> Path:
    extent = spatial_extent_mm(student)
    with torch.no_grad():
        exact_spatial = exact_output_intensity(student, powers)
        exact_detector = student(powers, method="exact", mode_chunk_size=4)
        realization_counts = np.asarray([1, 4, 16, 64, 256, 1024])
        detector_errors: list[float] = []
        for count in realization_counts:
            generator = torch.Generator(device=powers.device).manual_seed(100 + int(count))
            estimate = student(
                powers,
                method="stochastic",
                n_realizations=int(count),
                realization_chunk_size=64,
                mode_chunk_size=4,
                generator=generator,
            )
            detector_errors.append(float(relative_error(estimate, exact_detector)))
        stochastic_spatial = stochastic_output_intensity(
            student,
            powers,
            1024,
            seed=17,
        )
        generator = torch.Generator(device=powers.device).manual_seed(17)
        stochastic_detector = student(
            powers,
            method="stochastic",
            n_realizations=1024,
            realization_chunk_size=64,
            mode_chunk_size=4,
            generator=generator,
        )

    exact_np = as_numpy(exact_spatial)
    stochastic_np = as_numpy(stochastic_spatial)
    common_max = max(float(exact_np.max()), float(stochastic_np.max()))
    residual = (stochastic_np - exact_np) / max(common_max, 1e-30)
    residual_limit = max(float(np.max(np.abs(residual))), 1e-12)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    add_image(
        figure,
        axes[0, 0],
        exact_np / common_max,
        "Exact incoherent intensity",
        extent,
        vmin=0.0,
        vmax=1.0,
        colorbar_label="intensity / common max",
    )
    add_image(
        figure,
        axes[0, 1],
        stochastic_np / common_max,
        "Random-phase estimate (K=1024)",
        extent,
        vmin=0.0,
        vmax=1.0,
        colorbar_label="intensity / common max",
    )
    add_image(
        figure,
        axes[0, 2],
        residual,
        "Spatial residual / common max",
        extent,
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
        colorbar_label="normalized residual",
    )

    indices = np.arange(student.n_detectors)
    bar_width = 0.38
    axes[1, 0].bar(
        indices - bar_width / 2,
        as_numpy(exact_detector),
        bar_width,
        label="exact",
    )
    axes[1, 0].bar(
        indices + bar_width / 2,
        as_numpy(stochastic_detector),
        bar_width,
        label="stochastic",
    )
    axes[1, 0].set_title("Detector powers")
    axes[1, 0].set_xlabel("detector index")
    axes[1, 0].set_ylabel("power")
    axes[1, 0].set_xticks(indices)
    axes[1, 0].legend()

    axes[1, 1].loglog(
        realization_counts,
        detector_errors,
        "o-",
        color="#6a1b9a",
        label="measured error",
    )
    reference = detector_errors[0] / np.sqrt(realization_counts)
    axes[1, 1].loglog(
        realization_counts,
        reference,
        "--",
        color="gray",
        label=r"$K^{-1/2}$ reference",
    )
    axes[1, 1].set_title("Monte Carlo convergence")
    axes[1, 1].set_xlabel("number of realizations K")
    axes[1, 1].set_ylabel("detector relative error")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend()

    input_map = as_numpy(powers.reshape(2, 2))
    image = axes[1, 2].imshow(input_map, cmap="Blues", origin="lower")
    axes[1, 2].set_title("Input LED powers")
    axes[1, 2].set_xlabel("LED x index")
    axes[1, 2].set_ylabel("LED y index")
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_yticks([0, 1])
    for (row, col), value in np.ndenumerate(input_map):
        axes[1, 2].text(col, row, f"{value:.1f}", ha="center", va="center")
    figure.colorbar(image, ax=axes[1, 2], fraction=0.046, pad=0.04, label="relative power")

    figure.suptitle(
        "Exact mutual-intensity sum versus random-phase estimator",
        fontsize=15,
    )
    path = OUTPUT_DIR / "stochastic_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    teacher, student, target, fitted, losses = train_student(device)
    powers = torch.tensor([1.0, 0.3, 0.6, 0.1], device=device)

    paths = [
        plot_optical_fields(student, powers),
        plot_matrix_training(teacher, student, target, fitted, losses),
        plot_stochastic_comparison(student, powers),
    ]
    print(f"device={device} torch={torch.__version__}")
    print(f"matrix_relative_error={float(relative_error(fitted, target)):.6e}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
