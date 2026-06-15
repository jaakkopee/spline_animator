# Spline Animator (Sketch Repo)

A minimal, extendable Python project for experimenting with image-to-image spline interpolation and video rendering.

It includes:
- `birth_of_the_four_classical_elements.py`: generates the requested test images.
- `spline_animator.py`: a bare-bones but extendable spline interpolation pipeline with CLI subcommands.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Generate test images:

```bash
birth-of-elements --output-dir assets/test_images --width 640 --height 640
```

List keyframes:

```bash
spline-animator list-images --input-dir assets/test_images
```

Render a GIF using Catmull-Rom spline interpolation:

```bash
spline-animator render \
  --input-dir assets/test_images \
  --output artifacts/spline_demo.gif \
  --frames-per-segment 20 \
  --fps 24
```

Export interpolated frames for inspection:

```bash
spline-animator export-frames \
  --input-dir assets/test_images \
  --output-dir artifacts/frames \
  --frames-per-segment 20
```

## Notes

- This repo is intentionally small and easy to modify.
- `spline_animator.py` is structured so interpolation strategies can be replaced later.
- MP4 output is supported by `imageio` when FFmpeg is available.
