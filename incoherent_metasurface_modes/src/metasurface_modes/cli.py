"""Command-line interface for regular-array mode-overlap diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from .diagnostics import helstrom_intensity_columns, target_pair_feasibility
from .model import AngularModeArrayConfig, summarize_regular_array


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose non-orthogonality of MicroLED angular input modes."
    )
    parser.add_argument("--layout-x", type=int, default=256)
    parser.add_argument("--layout-y", type=int, default=192)
    parser.add_argument("--pitch-x-um", type=float, default=10.0)
    parser.add_argument("--pitch-y-um", type=float, default=10.0)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--focal-length-mm", type=float, default=20.0)
    parser.add_argument("--pupil", choices=("rectangle", "circle"), default="rectangle")
    parser.add_argument("--pupil-width-mm", type=float, default=1.064)
    parser.add_argument("--pupil-height-mm", type=float, default=1.064)
    parser.add_argument("--pupil-diameter-mm", type=float, default=1.064)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = AngularModeArrayConfig(
        layout_x=args.layout_x,
        layout_y=args.layout_y,
        led_pitch_x_um=args.pitch_x_um,
        led_pitch_y_um=args.pitch_y_um,
        wavelength_nm=args.wavelength_nm,
        focal_length_mm=args.focal_length_mm,
        pupil_shape=args.pupil,
        pupil_width_mm=args.pupil_width_mm,
        pupil_height_mm=args.pupil_height_mm,
        pupil_diameter_mm=args.pupil_diameter_mm,
    )
    summary = summarize_regular_array(config)
    if args.json:
        print(json.dumps({"config": asdict(config), "summary": asdict(summary)}, indent=2))
        return

    g = summary.max_off_diagonal_overlap
    optimal_a, optimal_b = helstrom_intensity_columns(g)
    optimal = target_pair_feasibility(g, optimal_a, optimal_b)
    disjoint = target_pair_feasibility(
        g,
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    )

    print("MicroLED input-mode overlap diagnostic")
    print("=" * 42)
    print(f"layout:                 {config.layout_x} x {config.layout_y} = {summary.n_led:,} LEDs")
    print(f"wavelength:             {config.wavelength_nm:.6g} nm")
    print(f"focal length:           {config.focal_length_mm:.6g} mm")
    print(f"LED pitch:              {config.led_pitch_x_um:.6g} x {config.led_pitch_y_um:.6g} um")
    if config.pupil_shape == "rectangle":
        print(f"pupil:                  {config.pupil_width_mm:.6g} x {config.pupil_height_mm:.6g} mm rectangle")
    else:
        print(f"pupil:                  {config.pupil_diameter_mm:.6g} mm diameter circle")
    print(f"angular step x/y:       {summary.angular_step_x_rad:.6g}, {summary.angular_step_y_rad:.6g} rad")
    print(f"edge angle x/y:         {np.degrees(summary.edge_angle_x_rad):.6g}, {np.degrees(summary.edge_angle_y_rad):.6g} deg")
    print()
    print(f"adjacent overlap x:     {summary.adjacent_x_overlap:.9g}")
    print(f"adjacent overlap y:     {summary.adjacent_y_overlap:.9g}")
    print(f"maximum off-diagonal:   {summary.max_off_diagonal_overlap:.9g}")
    print(f"maximum-overlap offset: ({summary.max_overlap_offset_x}, {summary.max_overlap_offset_y}) LED pitches")
    print(f"effective Gram rank:    {summary.effective_rank:,.3f} / {summary.n_led:,}")
    print(f"effective-rank ratio:   {summary.effective_rank_fraction:.9g}")
    print()
    print("Operational meaning of the worst pair")
    print("-" * 42)
    print(f"maximum output-column TV distance: {summary.worst_pair_max_total_variation:.9g}")
    print(f"optimal lossless measurement TV:   {optimal['total_variation']:.9g}")
    print(f"disjoint target columns feasible:  {disjoint['feasible']}")
    print()
    print("Defaults are illustrative. Replace layout and optical dimensions with the physical design.")


if __name__ == "__main__":
    main()

