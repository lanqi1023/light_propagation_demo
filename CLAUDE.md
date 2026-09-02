# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

An interactive web demo of the **Angular Spectrum Method (ASM)** for optical wave propagation. The computation core is a pure Python package (`lightprop/`) served via FastAPI; the frontend is a minimal HTML/JS shell that calls `/api/compute`.

## Commands

```bash
python server.py            # serve on http://localhost:8080
pytest tests/               # run unit tests
```

No build step; Python-only backend.

## Architecture

### Two-layer split

- **Python backend** ([lightprop/](lightprop/)): physics computation, pure CPU with numpy/scipy.
  - `types.py`: `Params` / `Result` dataclasses.
  - `aperture.py`: slit, double-slit, circle, rectangle, free, upload.
  - `angspec.py`: band-limited ASM + zero-padding.
  - `optics.py`: tilt, temporal/spatial coherence.
  - `fft.py`: 2D FFT via `scipy.fft`.
  - `pipeline.py`: orchestrator `compute(params) -> Result`.
- **FastAPI server** ([server.py](server.py)): serves static files and exposes `POST /api/compute`.
- **Frontend shell** ([templates/index.html](templates/index.html), [static/app_shell.js](static/app_shell.js), [static/colormap.js](static/colormap.js), [static/style.css](static/style.css)): UI bindings, canvas rendering, fetch to `/api/compute`.

## Physics pipeline

1. `Aperture.generate(...)` builds input field `U0` (shape `(Nx, Ny)` complex128).
2. `Optics.applyTilt(...)` optionally applies a plane-wave tilt.
3. `AngSpec.propagate(...)` propagates to distance `dz` via ASM with optional 2x zero-padding and Matsushima 2009 band-limiting.
4. Result post-processing computes intensity, phase, and center cross-sections.

## Critical conventions

- **Array layout**: numpy `(Nx, Ny)` complex128, row-major `(row=y, col=x)`.
- **Grid axes**: `Nx` = rows = y-dimension, `Ny` = cols = x-dimension.
- **Units**: `dx, dy` in μm; `lambda_nm` in nm; `dz` in mm; `Lx = Nx * dx` in μm.
- **Auto grid upgrade**: if band-limiting produces a degenerate result with <=4 unique intensity values, `pipeline.py` will automatically upgrade to a larger grid to preserve physical accuracy.

## Testing

pytest tests live in [tests/](tests/): `test_fft.py`, `test_aperture.py`, `test_angspec.py`.
