# Repository guidance

This repository contains one active project: a differentiable model of incoherent MicroLED illumination propagating through cascaded metasurfaces.

## Project layout

- `incoherent_metasurface_modes/`: source code, tests, examples, mathematical model, and generated figures.
- `docs/research/`: the complete discussion record, concise design conclusions, and the earlier modeling answer.
- `README.md`: repository entry point.

The former HTML/FastAPI angular-spectrum demo was intentionally removed. Do not restore or depend on `lightprop/`, `server.py`, `templates/`, `static/`, or the former root-level `tests/`.

## Environment and commands

Use the environment inside the active project when it exists:

```bash
cd incoherent_metasurface_modes
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python examples/simple_demo.py
PYTHONPATH=src .venv/bin/python examples/visualize_demo.py
PYTHONPATH=src .venv/bin/python examples/train_demo.py
```

To create a new environment:

```bash
cd incoherent_metasurface_modes
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[torch,test,viz]'
```

## Modeling conventions

- PyTorch complex tensors represent coherent fields for one source mode at a time.
- Mutually incoherent LED/sub-emitter modes are combined by summing detector intensities, never field amplitudes.
- The first surface is a rectangular-aperture collimating metalens with an optional trainable phase residual.
- Later surfaces are trainable phase-only masks separated by angular-spectrum propagation.
- Detector outputs are power integrals over non-overlapping regions.
- Preserve differentiability with respect to all trainable phase parameters.
- Keep exact incoherent mode summation as the reference model; stochastic random-phase estimation is an optional approximation.

See `incoherent_metasurface_modes/MATHEMATICAL_MODEL.md` before changing normalization, propagation, source sampling, or detector integration.
