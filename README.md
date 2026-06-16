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

Render with easing and alpha-aware blending options:

```bash
spline-animator render \
  --input-dir assets/test_images \
  --output artifacts/spline_eased.gif \
  --frames-per-segment 24 \
  --easing smoothstep \
  --interpolation catmull-rom \
  --alpha-blend premultiplied \
  --fps 24
```

Export interpolated frames for inspection:

```bash
spline-animator export-frames \
  --input-dir assets/test_images \
  --output-dir artifacts/frames \
  --frames-per-segment 20
```

Write a timeline template and render from JSON timeline:

```bash
spline-animator write-timeline-template --output timeline.example.json

spline-animator render \
  --input-dir assets/test_images \
  --timeline timeline.example.json \
  --output artifacts/spline_timeline.mp4 \
  --fps 24
```

## Timeline JSON format

Timeline lets you define keyframe order, per-segment duration, easing, interpolation mode, and alpha blending strategy.

```json
{
  "keyframes": [
    "01_red_point_blue_stars.png",
    "02_blue_circle_indigo_to_blue_alpha_outside.png",
    "03_blue_point_rainbow_spiral.png"
  ],
  "defaults": {
    "frames": 20,
    "easing": "smoothstep",
    "interpolation": "catmull-rom",
    "alpha_blend": "premultiplied"
  },
  "segments": [
    {
      "duration_frames": 12,
      "easing": "ease-in",
      "interpolation": "linear"
    },
    {
      "duration_seconds": 1.0,
      "easing": "ease-out",
      "alpha_blend": "straight"
    }
  ]
}
```

Supported easing values:
- `linear`
- `ease-in`
- `ease-out`
- `ease-in-out`
- `smoothstep`
- `smootherstep`

Supported interpolation values:
- `catmull-rom`
- `linear`

Supported alpha blending values:
- `premultiplied` (recommended for translucent assets)
- `straight`

## Notes

- This repo is intentionally small and easy to modify.
- `spline_animator.py` is structured so interpolation strategies can be replaced later.
- MP4 output is supported by `imageio` when FFmpeg is available.
