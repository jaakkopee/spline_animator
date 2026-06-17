# Audio-Reactive Timeline Generation Example

This document demonstrates the audio-reactive timeline generation feature with a complete end-to-end workflow.

## What is Audio-Reactive Timeline Generation?

The audio-reactive timeline generation system analyzes an audio file and automatically generates a spline animation timeline where:

1. **Timeline Duration** matches the audio file duration
2. **Segment Boundaries** are detected from audio onsets (beat-like transients)
3. **Segment Parameters** are dynamically determined by audio features:
   - Frame counts vary with loudness
   - Easing functions adapt to spectral content
   - Spline tension tracks energy levels
   - Interpolation endpoints respond to frequency brightness

This creates animations that naturally synchronize with and react to the audio's dynamics.

## Complete Workflow

### Step 1: Generate Test Assets

```bash
# Generate test images for keyframes
spline-animator generate-images --output-dir assets/test_images --width 640 --height 640

# Expected output:
# Generated 6 images:
# - 01_red_point_blue_stars.png
# - 02_blue_circle_indigo_to_blue_alpha_outside.png
# - 03_blue_point_rainbow_spiral.png
# - 04_darting_blue_line_rainbow_streaks.png
# - 05_violet_triangle_blue_fill.png
# - 06_square_multicolor_edges.png
```

### Step 2: Analyze Your Audio File

Before generating a timeline, inspect what the system detects in your audio:

```bash
spline-animator timeline-from-audio \
  --audio your_song.mp3 \
  --output /tmp/dummy.json \
  --analyze-only
```

This outputs:
```
Audio Analysis Report:
  Duration: 60.00s
  Sample Rate: 44100 Hz
  Energy range: [0.000, 1.000]
  Spectral Centroid range: [0.000, 1.000]
  Zero Crossing Rate range: [0.000, 1.000]
  Detected Onsets: 47
    Times: [0.069659 0.371519 0.766258 1.044897 ...]
```

### Step 3: Generate Timeline from Audio

Generate the timeline with detected segment boundaries and parameters:

```bash
spline-animator timeline-from-audio \
  --audio your_song.mp3 \
  --output timeline_audio_reactive.json \
  --fps 24 \
  --overwrite
```

Output shows:
```
Wrote audio-reactive timeline to timeline_audio_reactive.json
 - Audio file: your_song.mp3
 - Audio duration: 60.00s
 - Keyframes: 48 (auto-generated placeholders)
 - Segments: 47 (one per detected onset)
 - Estimated total frames: 1847
 - Estimated timeline duration: 76.96s at 24 fps
 - Detected onsets: 47
```

### Step 4: Using Specific Keyframes

If you have specific images you want to use, provide them:

```bash
spline-animator timeline-from-audio \
  --audio your_song.mp3 \
  --output timeline_with_images.json \
  --keyframes assets/test_images/01_red_point_blue_stars.png,assets/test_images/02_blue_circle_indigo_to_blue_alpha_outside.png,assets/test_images/03_blue_point_rainbow_spiral.png,assets/test_images/04_darting_blue_line_rainbow_streaks.png,assets/test_images/05_violet_triangle_blue_fill.png \
  --fps 24 \
  --overwrite
```

If you provide 5 keyframes but the audio has 47 detected onsets, the keyframes will cycle.

### Step 5: Validate the Timeline

Ensure the generated timeline is valid:

```bash
spline-animator validate-timeline \
  --timeline timeline_audio_reactive.json \
  --schema timeline.schema.json \
  --input-dir assets/test_images
```

### Step 6: Render the Animation

Create the final animation:

```bash
spline-animator render \
  --input-dir assets/test_images \
  --timeline timeline_audio_reactive.json \
  --output animation.gif \
  --fps 24
```

## How Parameters Are Determined

### Frame Count (Duration)
- **Default range**: 8-64 frames
- **Logic**: Scaled by energy - quiet sections animate fast, loud sections animate slowly

### Easing
- High energy + bright: `ease-out` (sharp transitions)
- Moderate energy: `ease-in-out` (smooth transitions)
- Low energy: `smoothstep` or `ease-in` (gentle transitions)

### Spline Tension
- **Range**: -1.0 (loose) to +1.0 (tight)
- **Logic**: Scaled by energy
- Low energy → negative tension (flowing curves)
- High energy → positive tension (controlled curves)

### Spline Endpoint
- Low energy (`< 0.6`): `clamp` (stable)
- High energy (`≥ 0.6`): `mirror` (smooth reflections)

## Supported Audio Formats

WAV, MP3, FLAC, OGG, M4A, AIFF, and more.

## Advanced Options

```bash
spline-animator timeline-from-audio \
  --audio song.mp3 \
  --output timeline.json \
  --fps 24 \
  --min-segment-frames 4 \
  --max-segment-frames 128 \
  --overwrite
```

All generated timelines work with existing timeline features:
- Schema validation
- Timeline doctor analysis
- Rendering with chroma key
- Further editing with timeline-wizard
