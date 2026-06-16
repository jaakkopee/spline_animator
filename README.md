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
  --spline-tension 0.15 \
  --spline-bias 0.0 \
  --spline-continuity 0.0 \
  --spline-endpoint clamp \
  --alpha-blend premultiplied \
  --chroma-key 0,0,0 \
  --chroma-threshold 18 \
  --fps 24
```

Render an MP4 from a custom image folder with stronger chroma keying:

```bash
spline-animator render \
  --input-dir ~/Documents/japipo_images \
  --output ~/Documents/japipo_images/japiposes3.mp4 \
  --frames-per-segment 64 \
  --easing ease-in-out \
  --interpolation catmull-rom \
  --spline-tension 0.25 \
  --spline-endpoint mirror \
  --chroma-key 0,0,0 \
  --chroma-threshold 128 \
  --mp4-background 8,12,24
```

Export interpolated frames for inspection:

```bash
spline-animator export-frames \
  --input-dir assets/test_images \
  --output-dir artifacts/frames \
  --frames-per-segment 20 \
  --chroma-key '#000000' \
  --chroma-threshold 16
```

Write a timeline template and render from JSON timeline:

```bash
spline-animator write-timeline-template --output timeline.example.json

spline-animator render \
  --input-dir assets/test_images \
  --timeline timeline.example.json \
  --output artifacts/spline_timeline.mp4 \
  --fps 24

spline-animator validate-timeline \
  --timeline timeline.example.json \
  --schema timeline.schema.json \
  --input-dir assets/test_images

spline-animator timeline-doctor \
  --timeline timeline.example.json \
  --schema timeline.schema.json \
  --input-dir assets/test_images \
  --fps 24

spline-animator timeline-wizard \
  --input-dir assets/test_images \
  --schema timeline.schema.json

spline-animator timeline-compose \
  --input-dir assets/test_images \
  --output timeline.compose.json \
  --preset cinematic-smooth \
  --frames-per-segment 24 \
  --keyframes 1,2,3,4 \
  --segment-overrides-file segment_overrides.json \
  --chroma-key 0,0,0 \
  --chroma-threshold 12 \
  --mp4-background 8,12,24 \
  --overwrite
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
    "alpha_blend": "premultiplied",
    "spline_tension": 0.1,
    "spline_bias": 0.0,
    "spline_continuity": -0.1,
    "spline_endpoint": "mirror"
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
      "alpha_blend": "straight",
      "spline_tension": 0.25,
      "spline_continuity": 0.1,
      "spline_endpoint": "wrap"
    }
  ]
}
```

Timeline spline fields:
- `spline_tension`, `spline_bias`, `spline_continuity` in `[-1, 1]`
- `spline_endpoint` in `{clamp, mirror, wrap}`
- You can place these in `defaults` and override per entry in `segments`.

Optional render hints in timeline JSON:
- `render_hints.chroma_key`
- `render_hints.chroma_threshold`
- `render_hints.mp4_background`

When rendering with `--timeline`, these hints are used automatically if equivalent CLI options are omitted.

Phase 1 timeline tooling:
- `timeline.schema.json`: JSON Schema for structural validation.
- `validate-timeline`: checks schema and runtime timeline resolution (paths, segment counts, coercion).
- `timeline-doctor`: prints resolved timeline settings, duration/frame estimates, and resource warnings.

Phase 2 timeline wizard:
- `timeline-wizard`: interactive terminal wizard for building/updating timeline JSON.
- Supports presets: `cinematic-smooth`, `stable-low-overshoot`, `loop-friendly`, `fast-cuts`.
- Uses mixed flow: auto-build segments from keyframes, then edit selected segments.
- Asks output mode each run: create new file or update existing file.
- Optional advanced section for chroma/MP4 render hints.
- Asks for confirmation on risky settings (high frame count / high estimated memory).

Phase 2 companion command:
- `timeline-compose`: non-interactive/scriptable timeline creation.
- Supports the same presets and spline controls as wizard defaults.
- `--keyframes` accepts comma-separated 1-based indices or file names/paths.
- `--segment-overrides-file` applies per-segment overrides for selected segments.
- `--overwrite` allows replacing existing output file.
- `--allow-risky` bypasses safety stop for very large frame/memory estimates.

Example `segment_overrides.json`:

```json
[
  {
    "index": 2,
    "duration_frames": 12,
    "easing": "ease-in",
    "interpolation": "linear"
  },
  {
    "index": 3,
    "duration_seconds": 1.2,
    "spline_tension": 0.3,
    "spline_endpoint": "wrap"
  }
]
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

Spline shape controls (for `catmull-rom` interpolation):
- `--spline-tension` in `[-1, 1]`
- `--spline-bias` in `[-1, 1]`
- `--spline-continuity` in `[-1, 1]`
- `--spline-endpoint` in `{clamp, mirror, wrap}`

How these affect motion:
- `tension`:
  - `0.0` is classic Catmull-Rom.
  - Higher values (e.g. `0.2..0.6`) tighten curves and reduce overshoot.
  - Lower values (e.g. negative) make arcs looser and can increase overshoot.
- `bias`:
  - Positive values bias movement toward the incoming side of each keyframe.
  - Negative values bias movement toward the outgoing side.
  - `0.0` is symmetric.
- `continuity`:
  - Positive values make transitions sharper at keyframes.
  - Negative values smooth them out.
  - `0.0` is neutral.
- `endpoint`:
  - `clamp`: duplicates edge points (stable default).
  - `mirror`: reflects edge tangents, often smoother starts/ends.
  - `wrap`: treats sequence as looped (good for cyclic animations).

Starter presets:
- Stable/noisy-data friendly: `--spline-tension 0.35 --spline-bias 0 --spline-continuity 0 --spline-endpoint clamp`
- Smooth cinematic: `--spline-tension 0.1 --spline-bias 0 --spline-continuity -0.15 --spline-endpoint mirror`
- Looped motion: `--spline-tension 0 --spline-bias 0 --spline-continuity 0 --spline-endpoint wrap`

Supported alpha blending values:
- `premultiplied` (recommended for translucent assets)
- `straight`

Chroma key options:
- `--chroma-key`: color to make transparent, format `R,G,B` or `#RRGGBB`
- `--chroma-threshold`: RGB distance threshold (default `0.0`, exact match). Larger values remove a wider color neighborhood.

MP4 output options:
- `--mp4-background`: color used to composite transparent pixels before MP4 encoding, format `R,G,B` or `#RRGGBB`.

Important output note:
- PNG frame export preserves alpha.
- GIF supports transparency in palette-based form.
- MP4 output is RGB only (alpha is discarded by codec/container), so use `--mp4-background` if you want transparent areas to become a specific visible color.

## Notes

- This repo is intentionally small and easy to modify.
- `spline_animator.py` is structured so interpolation strategies can be replaced later.
- MP4 output is supported by `imageio` when FFmpeg is available.
