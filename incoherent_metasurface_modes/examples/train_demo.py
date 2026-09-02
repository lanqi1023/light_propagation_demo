"""Small end-to-end optimization demo for the incoherent PyTorch model."""

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


def build_demo_model(device: torch.device) -> IncoherentMetasurfaceModel:
    wavelength_m = 532e-9
    focal_length_m = 20e-3
    grid = CartesianGrid(
        height=192,
        width=216,
        spacing_y_m=6e-6,
        spacing_x_m=6e-6,
    )
    aperture = rectangular_aperture(
        grid,
        height_m=1.0e-3,
        width_m=1.2e-3,
        device=device,
    )
    source = LambertianLEDArray(
        LambertianLEDArrayConfig(
            layout_y=4,
            layout_x=4,
            pitch_y_m=30e-6,
            pitch_x_m=30e-6,
            source_to_metasurface_m=focal_length_m,
            emitter_height_m=4e-6,
            emitter_width_m=4e-6,
            emitter_samples_y=1,
            emitter_samples_x=1,
        ),
        aperture=aperture,
    )
    optical_system = MultilayerMetasurface(
        grid,
        wavelength_m,
        focal_length_m,
        n_trainable_layers=2,
        propagation_distances_m=(5e-3, 5e-3, 5e-3),
        aperture=aperture,
        trainable_collimator_residual=False,
    )
    detector = AreaDetector(
        tiled_detector_masks(grid, 4, 4, device=device)
    )
    return IncoherentMetasurfaceModel(source, optical_system, detector).to(device)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_demo_model(device)

    # Construct a feasible-scale non-negative target for demonstration.  This
    # is not a claim that an arbitrary target is realizable by two phase layers.
    with torch.no_grad():
        initial_matrix = model.intensity_matrix(mode_chunk_size=8)
        mean_collected_power = initial_matrix.sum(dim=0).mean()
        target_matrix = torch.rand(
            model.n_detectors,
            model.n_leds,
            device=device,
        )
        target_matrix = target_matrix / target_matrix.sum(dim=0, keepdim=True)
        target_matrix = 0.6 * mean_collected_power * target_matrix

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    for step in range(101):
        led_powers = torch.rand(8, model.n_leds, device=device)
        target = led_powers @ target_matrix.transpose(0, 1)
        prediction = model(
            led_powers,
            method="stochastic",
            n_realizations=8,
            mode_chunk_size=8,
        )
        # Normalize by the target scale only to make the printed value readable.
        scale = target.square().mean().detach().clamp_min(1e-16)
        loss = (prediction - target).square().mean() / scale

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"step={step:03d} normalized_mse={loss.item():.6f}")

    with torch.no_grad():
        fitted_matrix = model.intensity_matrix(mode_chunk_size=8)
        relative_error = (
            torch.linalg.vector_norm(fitted_matrix - target_matrix)
            / torch.linalg.vector_norm(target_matrix)
        )
        print(f"matrix_relative_error={relative_error.item():.6f}")


if __name__ == "__main__":
    main()
