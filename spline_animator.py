from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image

from birth_of_the_four_classical_elements import generate_test_images


@dataclass
class RenderConfig:
    fps: int = 24
    mp4_background: tuple[int, int, int] | None = None


@dataclass
class ChromaKeySpec:
    color: tuple[int, int, int]
    threshold: float = 0.0


@dataclass
class SplineControl:
    tension: float = 0.0
    bias: float = 0.0
    continuity: float = 0.0
    endpoint: str = "clamp"


@dataclass
class SegmentSpec:
    frames: int
    easing: str = "linear"
    interpolation: str = "catmull-rom"
    alpha_blend: str = "premultiplied"
    spline_tension: float = 0.0
    spline_bias: float = 0.0
    spline_continuity: float = 0.0
    spline_endpoint: str = "clamp"


WIZARD_PRESETS: dict[str, SegmentSpec] = {
    "cinematic-smooth": SegmentSpec(
        frames=32,
        easing="ease-in-out",
        interpolation="catmull-rom",
        alpha_blend="premultiplied",
        spline_tension=0.1,
        spline_bias=0.0,
        spline_continuity=-0.15,
        spline_endpoint="mirror",
    ),
    "stable-low-overshoot": SegmentSpec(
        frames=20,
        easing="smoothstep",
        interpolation="catmull-rom",
        alpha_blend="premultiplied",
        spline_tension=0.35,
        spline_bias=0.0,
        spline_continuity=0.0,
        spline_endpoint="clamp",
    ),
    "loop-friendly": SegmentSpec(
        frames=24,
        easing="linear",
        interpolation="catmull-rom",
        alpha_blend="premultiplied",
        spline_tension=0.0,
        spline_bias=0.0,
        spline_continuity=0.0,
        spline_endpoint="wrap",
    ),
    "fast-cuts": SegmentSpec(
        frames=8,
        easing="ease-out",
        interpolation="linear",
        alpha_blend="straight",
        spline_tension=0.0,
        spline_bias=0.0,
        spline_continuity=0.0,
        spline_endpoint="clamp",
    ),
}


class SplineAnimator:
    """Minimal Catmull-Rom image spline interpolator."""

    def __init__(self, keyframes: list[np.ndarray]):
        if len(keyframes) < 2:
            raise ValueError("Need at least two keyframes.")
        self.keyframes = [frame.astype(np.float32) for frame in keyframes]
        self.keyframes_premultiplied = [self._to_premultiplied(frame) for frame in self.keyframes]

    @classmethod
    def from_paths(cls, image_paths: list[Path]) -> "SplineAnimator":
        if len(image_paths) < 2:
            raise ValueError("Need at least two images to interpolate.")

        images = [Image.open(path).convert("RGBA") for path in image_paths]
        base_size = images[0].size

        normalized: list[np.ndarray] = []
        for image in images:
            if image.size != base_size:
                image = image.resize(base_size, Image.Resampling.LANCZOS)
            normalized.append(np.array(image, dtype=np.float32))
        return cls(normalized)

    @staticmethod
    def _catmull_rom(
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        t: float,
    ) -> np.ndarray:
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    @classmethod
    def _kochanek_bartels(
        cls,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        t: float,
        spline: SplineControl,
    ) -> np.ndarray:
        if spline.tension == 0.0 and spline.bias == 0.0 and spline.continuity == 0.0:
            return cls._catmull_rom(p0, p1, p2, p3, t)

        tension = spline.tension
        bias = spline.bias
        continuity = spline.continuity

        m1 = (
            0.5 * (1.0 - tension) * (1.0 + bias) * (1.0 + continuity) * (p1 - p0)
            + 0.5 * (1.0 - tension) * (1.0 - bias) * (1.0 - continuity) * (p2 - p1)
        )
        m2 = (
            0.5 * (1.0 - tension) * (1.0 + bias) * (1.0 - continuity) * (p2 - p1)
            + 0.5 * (1.0 - tension) * (1.0 - bias) * (1.0 + continuity) * (p3 - p2)
        )

        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        return h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2

    @staticmethod
    def _linear(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
        return p1 + (p2 - p1) * t

    @staticmethod
    def _to_premultiplied(frame: np.ndarray) -> np.ndarray:
        alpha = frame[..., 3:4] / 255.0
        premult_rgb = frame[..., :3] * alpha
        return np.concatenate((premult_rgb, frame[..., 3:4]), axis=-1)

    @staticmethod
    def _from_premultiplied(frame: np.ndarray) -> np.ndarray:
        alpha = frame[..., 3:4]
        safe_alpha = np.where(alpha > 1e-6, alpha, 1.0)
        rgb = frame[..., :3] * (255.0 / safe_alpha)
        rgb = np.where(alpha > 1e-6, rgb, 0.0)
        return np.concatenate((rgb, alpha), axis=-1)

    @staticmethod
    def _normalize_easing(name: str) -> str:
        return name.strip().lower().replace("_", "-")

    @staticmethod
    def _normalize_interpolation(name: str) -> str:
        return name.strip().lower().replace("_", "-")

    @staticmethod
    def _normalize_alpha_blend(name: str) -> str:
        return name.strip().lower().replace("_", "-")

    @staticmethod
    def _normalize_endpoint(name: str) -> str:
        return name.strip().lower().replace("_", "-")

    @staticmethod
    def _apply_chroma_key(frame: np.ndarray, chroma_key: ChromaKeySpec | None) -> np.ndarray:
        if chroma_key is None:
            return frame

        key_rgb = np.array(chroma_key.color, dtype=np.float32)
        rgb = frame[..., :3].astype(np.float32)
        dist = np.linalg.norm(rgb - key_rgb, axis=-1)
        mask = dist <= float(chroma_key.threshold)

        output = frame.copy()
        output[..., 3] = np.where(mask, 0, output[..., 3])
        return output

    @staticmethod
    def _composite_over_background(frame: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
        rgb = frame[..., :3].astype(np.float32)
        alpha = frame[..., 3:4].astype(np.float32) / 255.0
        bg = np.array(background, dtype=np.float32).reshape((1, 1, 3))
        composite = rgb * alpha + bg * (1.0 - alpha)
        return np.clip(composite, 0, 255).astype(np.uint8)

    @classmethod
    def _apply_easing(cls, t: float, easing: str) -> float:
        e = cls._normalize_easing(easing)
        if e == "linear":
            return t
        if e in {"ease-in", "ease-in-quad"}:
            return t * t
        if e in {"ease-out", "ease-out-quad"}:
            return 1.0 - (1.0 - t) * (1.0 - t)
        if e in {"ease-in-out", "ease-in-out-quad"}:
            if t < 0.5:
                return 2.0 * t * t
            return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0
        if e == "smoothstep":
            return t * t * (3.0 - 2.0 * t)
        if e == "smootherstep":
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        raise ValueError(f"Unsupported easing '{easing}'")

    @classmethod
    def _interpolate_arrays(
        cls,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        t: float,
        interpolation: str,
        spline: SplineControl,
    ) -> np.ndarray:
        mode = cls._normalize_interpolation(interpolation)
        if mode in {"catmull-rom", "catmullrom"}:
            return cls._kochanek_bartels(p0, p1, p2, p3, t, spline)
        if mode == "linear":
            return cls._linear(p1, p2, t)
        raise ValueError(f"Unsupported interpolation mode '{interpolation}'")

    @classmethod
    def _segment_points(
        cls,
        source: list[np.ndarray],
        segment_index: int,
        endpoint_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        i = segment_index
        n = len(source)

        p1 = source[i]
        p2 = source[i + 1]
        endpoint = cls._normalize_endpoint(endpoint_mode)

        if endpoint == "clamp":
            p0 = source[i - 1] if i > 0 else source[0]
            p3 = source[i + 2] if i + 2 < n else source[-1]
            return p0, p1, p2, p3

        if endpoint == "wrap":
            p0 = source[(i - 1) % n]
            p3 = source[(i + 2) % n]
            return p0, p1, p2, p3

        if endpoint == "mirror":
            if i > 0:
                p0 = source[i - 1]
            else:
                p0 = 2.0 * source[0] - source[1]

            if i + 2 < n:
                p3 = source[i + 2]
            else:
                p3 = 2.0 * source[-1] - source[-2]
            return p0, p1, p2, p3

        raise ValueError(f"Unsupported endpoint mode '{endpoint_mode}'")

    def _interpolate_segment_frame(
        self,
        segment_index: int,
        t: float,
        spec: SegmentSpec,
    ) -> np.ndarray:
        i = segment_index
        spline = SplineControl(
            tension=spec.spline_tension,
            bias=spec.spline_bias,
            continuity=spec.spline_continuity,
            endpoint=spec.spline_endpoint,
        )
        blend_mode = self._normalize_alpha_blend(spec.alpha_blend)

        if blend_mode == "straight":
            source = self.keyframes
            convert_back = False
        elif blend_mode in {"premultiplied", "premult"}:
            source = self.keyframes_premultiplied
            convert_back = True
        else:
            raise ValueError(f"Unsupported alpha blend mode '{spec.alpha_blend}'")

        p0, p1, p2, p3 = self._segment_points(source, i, spline.endpoint)

        eased_t = self._apply_easing(t, spec.easing)
        frame = self._interpolate_arrays(p0, p1, p2, p3, eased_t, spec.interpolation, spline)
        if convert_back:
            frame = self._from_premultiplied(frame)
        return frame

    def interpolated_frames(
        self,
        segments: list[SegmentSpec],
        chroma_key: ChromaKeySpec | None = None,
    ) -> list[np.ndarray]:
        if len(segments) != len(self.keyframes) - 1:
            raise ValueError("Number of segment specs must match number of keyframe transitions.")
        if any(spec.frames < 1 for spec in segments):
            raise ValueError("Each segment must have at least one frame.")

        frames: list[np.ndarray] = []
        for i, spec in enumerate(segments):
            for step in range(spec.frames):
                t = step / spec.frames
                frame = self._interpolate_segment_frame(i, t, spec)
                frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(self._apply_chroma_key(frame_u8, chroma_key))

        final_frame = np.clip(self.keyframes[-1], 0, 255).astype(np.uint8)
        frames.append(self._apply_chroma_key(final_frame, chroma_key))
        return frames

    def export_frames(
        self,
        output_dir: Path,
        segments: list[SegmentSpec],
        chroma_key: ChromaKeySpec | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.interpolated_frames(segments, chroma_key=chroma_key)

        written: list[Path] = []
        for idx, frame in enumerate(frames):
            path = output_dir / f"frame_{idx:04d}.png"
            Image.fromarray(frame, mode="RGBA").save(path)
            written.append(path)
        return written

    def render_video(
        self,
        output_path: Path,
        config: RenderConfig,
        segments: list[SegmentSpec],
        chroma_key: ChromaKeySpec | None = None,
    ) -> None:
        frames = self.interpolated_frames(segments, chroma_key=chroma_key)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            iio.imwrite(output_path, frames, format="GIF", duration=1.0 / config.fps, loop=0)
            return

        if suffix == ".mp4":
            if config.mp4_background is None:
                rgb_frames = [frame[..., :3] for frame in frames]
            else:
                rgb_frames = [self._composite_over_background(frame, config.mp4_background) for frame in frames]
            iio.imwrite(output_path, rgb_frames, fps=config.fps, codec="libx264")
            return

        raise ValueError("Unsupported output format. Use .gif or .mp4")


def discover_images(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.glob("*.png") if p.is_file()])


def _prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("Please enter a value.")


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    options = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{options}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _prompt_int(prompt: str, default: int, min_value: int | None = None) -> int:
    while True:
        raw = _prompt_text(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return value


def _prompt_float(prompt: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    while True:
        raw = _prompt_text(prompt, str(default))
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}.")
            continue
        return value


def _prompt_choice(prompt: str, options: list[str], default: str) -> str:
    option_set = set(options)
    while True:
        value = _prompt_text(prompt, default).strip().lower()
        if value in option_set:
            return value
        print(f"Choose one of: {', '.join(options)}")


def _default_segment_spec_from_args(args: argparse.Namespace) -> SegmentSpec:
    spline = _spline_control_from_args(args)
    return SegmentSpec(
        frames=args.frames_per_segment,
        easing=args.easing,
        interpolation=args.interpolation,
        alpha_blend=args.alpha_blend,
        spline_tension=spline.tension,
        spline_bias=spline.bias,
        spline_continuity=spline.continuity,
        spline_endpoint=spline.endpoint,
    )


def _parse_rgb_color(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        hex_value = text[1:]
        if len(hex_value) != 6:
            raise ValueError("Hex chroma key must be in #RRGGBB format.")
        try:
            return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError as exc:
            raise ValueError("Invalid hex chroma key value.") from exc

    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Chroma key color must be R,G,B or #RRGGBB.")

    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError("Chroma key RGB channels must be integers.") from exc

    if any(ch < 0 or ch > 255 for ch in rgb):
        raise ValueError("Chroma key RGB channels must be in range 0-255.")
    return rgb  # type: ignore[return-value]


def _chroma_key_from_args(args: argparse.Namespace) -> ChromaKeySpec | None:
    raw = getattr(args, "chroma_key", None)
    if not raw:
        return None
    color = _parse_rgb_color(raw)
    threshold = float(getattr(args, "chroma_threshold", 0.0))
    if threshold < 0:
        raise ValueError("chroma-threshold must be >= 0")
    return ChromaKeySpec(color=color, threshold=threshold)


def _add_chroma_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chroma-key",
        default=None,
        help="Set pixels near this color transparent. Format: R,G,B or #RRGGBB",
    )
    parser.add_argument(
        "--chroma-threshold",
        type=float,
        default=0.0,
        help="Euclidean RGB distance threshold for chroma keying.",
    )


def _add_mp4_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mp4-background",
        default=None,
        help="Composite RGBA over this color for MP4 output. Format: R,G,B or #RRGGBB",
    )


def _add_spline_control_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spline-tension",
        type=float,
        default=0.0,
        help="Spline tension in [-1,1]. 0=Catmull-Rom, higher values reduce overshoot.",
    )
    parser.add_argument(
        "--spline-bias",
        type=float,
        default=0.0,
        help="Spline bias in [-1,1]. Positive favors incoming tangent, negative favors outgoing.",
    )
    parser.add_argument(
        "--spline-continuity",
        type=float,
        default=0.0,
        help="Spline continuity in [-1,1]. Controls corner sharpness around keyframes.",
    )
    parser.add_argument(
        "--spline-endpoint",
        default="clamp",
        choices=["clamp", "mirror", "wrap"],
        help="Endpoint handling for Catmull-Rom style interpolation.",
    )


def _add_timeline_resolution_defaults(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames-per-segment", type=int, default=20)
    parser.add_argument(
        "--easing",
        default="linear",
        choices=["linear", "ease-in", "ease-out", "ease-in-out", "smoothstep", "smootherstep"],
    )
    parser.add_argument("--interpolation", default="catmull-rom", choices=["catmull-rom", "linear"])
    parser.add_argument("--alpha-blend", default="premultiplied", choices=["straight", "premultiplied"])
    parser.add_argument("--fps", type=int, default=24)
    _add_spline_control_args(parser)


def _normalize_endpoint_value(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _coerce_spline_control(
    tension_raw: Any,
    bias_raw: Any,
    continuity_raw: Any,
    endpoint_raw: Any,
    context: str,
) -> SplineControl:
    try:
        tension = float(tension_raw)
        bias = float(bias_raw)
        continuity = float(continuity_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid spline control value in {context}") from exc

    for name, value in (("spline_tension", tension), ("spline_bias", bias), ("spline_continuity", continuity)):
        if value < -1.0 or value > 1.0:
            raise ValueError(f"{name} in {context} must be in range [-1, 1]")

    endpoint = _normalize_endpoint_value(str(endpoint_raw))
    if endpoint not in {"clamp", "mirror", "wrap"}:
        raise ValueError(f"spline_endpoint in {context} must be one of clamp, mirror, wrap")

    return SplineControl(tension=tension, bias=bias, continuity=continuity, endpoint=endpoint)


def _spline_control_from_args(args: argparse.Namespace) -> SplineControl:
    return _coerce_spline_control(
        tension_raw=getattr(args, "spline_tension", 0.0),
        bias_raw=getattr(args, "spline_bias", 0.0),
        continuity_raw=getattr(args, "spline_continuity", 0.0),
        endpoint_raw=getattr(args, "spline_endpoint", "clamp"),
        context="CLI arguments",
    )


def _resolve_keyframe_path(name: str, input_dir: Path) -> Path:
    path = Path(name)
    return path if path.is_absolute() else (input_dir / path)


def _segments_for_keyframes(
    keyframe_count: int,
    timeline_segments: list[dict[str, Any]] | None,
    default_spec: SegmentSpec,
    fps: int,
) -> list[SegmentSpec]:
    required = keyframe_count - 1
    if required < 1:
        raise ValueError("Need at least two keyframes in timeline.")

    if timeline_segments is None:
        return [SegmentSpec(**vars(default_spec)) for _ in range(required)]

    if len(timeline_segments) != required:
        raise ValueError(
            f"Timeline has {required} transitions, but segments has {len(timeline_segments)} entries."
        )

    specs: list[SegmentSpec] = []
    for idx, raw in enumerate(timeline_segments):
        duration_frames = raw.get("frames", raw.get("duration_frames"))
        duration_seconds = raw.get("duration_seconds")

        if duration_frames is None and duration_seconds is not None:
            duration_frames = max(1, int(round(float(duration_seconds) * fps)))
        if duration_frames is None:
            duration_frames = default_spec.frames

        try:
            frames = int(duration_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid duration in timeline segment {idx}") from exc

        spline = _coerce_spline_control(
            tension_raw=raw.get("spline_tension", raw.get("tension", default_spec.spline_tension)),
            bias_raw=raw.get("spline_bias", raw.get("bias", default_spec.spline_bias)),
            continuity_raw=raw.get(
                "spline_continuity",
                raw.get("continuity", default_spec.spline_continuity),
            ),
            endpoint_raw=raw.get("spline_endpoint", raw.get("endpoint", default_spec.spline_endpoint)),
            context=f"timeline segment {idx}",
        )

        spec = SegmentSpec(
            frames=frames,
            easing=str(raw.get("easing", default_spec.easing)),
            interpolation=str(raw.get("interpolation", default_spec.interpolation)),
            alpha_blend=str(raw.get("alpha_blend", default_spec.alpha_blend)),
            spline_tension=spline.tension,
            spline_bias=spline.bias,
            spline_continuity=spline.continuity,
            spline_endpoint=spline.endpoint,
        )
        specs.append(spec)

    return specs


def _load_timeline(
    timeline_path: Path,
    input_dir: Path,
    default_spec: SegmentSpec,
    fps: int,
) -> tuple[list[Path], list[SegmentSpec]]:
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Timeline JSON root must be an object.")

    keyframes_data = data.get("keyframes")
    if not isinstance(keyframes_data, list) or len(keyframes_data) < 2:
        raise ValueError("Timeline requires 'keyframes' list with at least 2 entries.")

    keyframe_paths: list[Path] = []
    for idx, item in enumerate(keyframes_data):
        if not isinstance(item, str):
            raise ValueError(f"Timeline keyframe {idx} must be a string path.")
        keyframe_path = _resolve_keyframe_path(item, input_dir)
        if not keyframe_path.exists():
            raise ValueError(f"Timeline keyframe not found: {keyframe_path}")
        keyframe_paths.append(keyframe_path)

    raw_defaults = data.get("defaults", {})
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise ValueError("Timeline 'defaults' must be an object when provided.")

    default_spline = _coerce_spline_control(
        tension_raw=raw_defaults.get("spline_tension", raw_defaults.get("tension", default_spec.spline_tension)),
        bias_raw=raw_defaults.get("spline_bias", raw_defaults.get("bias", default_spec.spline_bias)),
        continuity_raw=raw_defaults.get(
            "spline_continuity",
            raw_defaults.get("continuity", default_spec.spline_continuity),
        ),
        endpoint_raw=raw_defaults.get("spline_endpoint", raw_defaults.get("endpoint", default_spec.spline_endpoint)),
        context="timeline defaults",
    )

    merged_default = SegmentSpec(
        frames=int(raw_defaults.get("frames", default_spec.frames)),
        easing=str(raw_defaults.get("easing", default_spec.easing)),
        interpolation=str(raw_defaults.get("interpolation", default_spec.interpolation)),
        alpha_blend=str(raw_defaults.get("alpha_blend", default_spec.alpha_blend)),
        spline_tension=default_spline.tension,
        spline_bias=default_spline.bias,
        spline_continuity=default_spline.continuity,
        spline_endpoint=default_spline.endpoint,
    )

    raw_segments = data.get("segments")
    if raw_segments is not None and not isinstance(raw_segments, list):
        raise ValueError("Timeline 'segments' must be a list when provided.")

    segments = _segments_for_keyframes(
        keyframe_count=len(keyframe_paths),
        timeline_segments=raw_segments,
        default_spec=merged_default,
        fps=fps,
    )

    return keyframe_paths, segments


def _resolve_image_paths_and_segments(
    input_dir: Path,
    timeline_path: Path | None,
    default_spec: SegmentSpec,
    fps: int,
) -> tuple[list[Path], list[SegmentSpec]]:
    if timeline_path is not None:
        image_paths, segments = _load_timeline(
            timeline_path=timeline_path,
            input_dir=input_dir,
            default_spec=default_spec,
            fps=fps,
        )
    else:
        image_paths = discover_images(input_dir)
        segments = _segments_for_keyframes(
            keyframe_count=len(image_paths),
            timeline_segments=None,
            default_spec=default_spec,
            fps=fps,
        )

    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 keyframes from {input_dir}")
    return image_paths, segments


def _load_animator_and_segments(
    input_dir: Path,
    timeline_path: Path | None,
    default_spec: SegmentSpec,
    fps: int,
) -> tuple[SplineAnimator, list[Path], list[SegmentSpec]]:
    image_paths, segments = _resolve_image_paths_and_segments(
        input_dir=input_dir,
        timeline_path=timeline_path,
        default_spec=default_spec,
        fps=fps,
    )

    return SplineAnimator.from_paths(image_paths), image_paths, segments


def _format_json_error_path(path_tokens: list[Any]) -> str:
    if not path_tokens:
        return "$"
    path = "$"
    for token in path_tokens:
        if isinstance(token, int):
            path += f"[{token}]"
        else:
            path += f".{token}"
    return path


def _validate_timeline_schema(timeline_path: Path, schema_path: Path) -> list[str]:
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("jsonschema package is required for schema validation") from exc

    timeline_data = json.loads(timeline_path.read_text(encoding="utf-8"))
    schema_data = json.loads(schema_path.read_text(encoding="utf-8"))

    validator = jsonschema.Draft202012Validator(schema_data)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(timeline_data), key=lambda e: list(e.path)):
        path = _format_json_error_path(list(error.path))
        errors.append(f"{path}: {error.message}")
    return errors


def _load_render_hints_from_timeline(timeline_path: Path) -> tuple[ChromaKeySpec | None, tuple[int, int, int] | None]:
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    raw_hints = data.get("render_hints")
    if raw_hints is None:
        return None, None
    if not isinstance(raw_hints, dict):
        raise ValueError("render_hints must be an object when provided")

    chroma: ChromaKeySpec | None = None
    background: tuple[int, int, int] | None = None

    raw_chroma = raw_hints.get("chroma_key")
    if raw_chroma is not None:
        color = _parse_rgb_color(str(raw_chroma))
        threshold = float(raw_hints.get("chroma_threshold", 0.0))
        if threshold < 0:
            raise ValueError("render_hints.chroma_threshold must be >= 0")
        chroma = ChromaKeySpec(color=color, threshold=threshold)

    raw_bg = raw_hints.get("mp4_background")
    if raw_bg is not None:
        background = _parse_rgb_color(str(raw_bg))

    return chroma, background


def _estimate_rgba_frame_buffer_gib(width: int, height: int, frames: int) -> float:
    byte_count = width * height * 4 * frames
    return byte_count / (1024.0**3)


def _timeline_doctor_warnings(
    image_paths: list[Path],
    segments: list[SegmentSpec],
    fps: int,
) -> tuple[list[str], tuple[int, int], float, int]:
    warnings: list[str] = []
    sizes: list[tuple[int, int]] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            sizes.append(image.size)

    unique_sizes = sorted(set(sizes))
    max_size = max(unique_sizes, key=lambda s: s[0] * s[1])
    if len(unique_sizes) > 1:
        warnings.append("Keyframes have mixed resolutions and will be resized to first image size.")

    total_frames = sum(segment.frames for segment in segments) + 1
    approx_gib = _estimate_rgba_frame_buffer_gib(max_size[0], max_size[1], total_frames)
    if total_frames >= 2000:
        warnings.append("High frame count detected; render may be slow and memory intensive.")
    if approx_gib >= 1.0:
        warnings.append(
            "Estimated uncompressed RGBA frame buffer exceeds 1 GiB; consider fewer frames per segment."
        )

    for idx, segment in enumerate(segments):
        has_custom_spline = any(
            [
                segment.spline_tension != 0.0,
                segment.spline_bias != 0.0,
                segment.spline_continuity != 0.0,
                segment.spline_endpoint != "clamp",
            ]
        )
        if segment.interpolation == "linear" and has_custom_spline:
            warnings.append(f"Segment {idx} uses linear interpolation; spline_* values are ignored.")

    if fps <= 0:
        warnings.append("FPS should be positive for duration estimates.")

    return warnings, max_size, approx_gib, total_frames


def _segment_to_json_dict(segment: SegmentSpec) -> dict[str, Any]:
    return {
        "duration_frames": segment.frames,
        "easing": segment.easing,
        "interpolation": segment.interpolation,
        "alpha_blend": segment.alpha_blend,
        "spline_tension": segment.spline_tension,
        "spline_bias": segment.spline_bias,
        "spline_continuity": segment.spline_continuity,
        "spline_endpoint": segment.spline_endpoint,
    }


def _wizard_pick_keyframes(available: list[Path]) -> list[Path]:
    print("Available keyframes:")
    for idx, image in enumerate(available, start=1):
        print(f" {idx}. {image.name}")

    if _prompt_yes_no("Use discovered order as-is?", default=True):
        return available

    while True:
        raw = _prompt_text("Enter comma-separated 1-based indices", "1,2")
        try:
            indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
        except ValueError:
            print("Please enter integers like 1,2,3")
            continue

        if len(indices) < 2:
            print("Please choose at least two keyframes.")
            continue
        if any(i < 1 or i > len(available) for i in indices):
            print("One or more indices are out of range.")
            continue
        selected = [available[i - 1] for i in indices]
        return selected


def _wizard_select_preset() -> tuple[str, SegmentSpec]:
    presets = list(WIZARD_PRESETS.keys())
    print("Presets:")
    for idx, name in enumerate(presets, start=1):
        print(f" {idx}. {name}")

    while True:
        raw = _prompt_text("Choose preset number", "1")
        try:
            index = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= index <= len(presets):
            name = presets[index - 1]
            return name, SegmentSpec(**vars(WIZARD_PRESETS[name]))
        print("Preset index out of range.")


def _wizard_edit_segment(segment: SegmentSpec, segment_index: int) -> SegmentSpec:
    print(f"Editing segment {segment_index}")
    segment.frames = _prompt_int("  duration_frames", segment.frames, min_value=1)
    segment.easing = _prompt_choice(
        "  easing",
        ["linear", "ease-in", "ease-out", "ease-in-out", "smoothstep", "smootherstep"],
        segment.easing,
    )
    segment.interpolation = _prompt_choice("  interpolation", ["catmull-rom", "linear"], segment.interpolation)
    segment.alpha_blend = _prompt_choice("  alpha_blend", ["premultiplied", "straight"], segment.alpha_blend)
    segment.spline_tension = _prompt_float("  spline_tension", segment.spline_tension, min_value=-1.0, max_value=1.0)
    segment.spline_bias = _prompt_float("  spline_bias", segment.spline_bias, min_value=-1.0, max_value=1.0)
    segment.spline_continuity = _prompt_float(
        "  spline_continuity", segment.spline_continuity, min_value=-1.0, max_value=1.0
    )
    segment.spline_endpoint = _prompt_choice(
        "  spline_endpoint", ["clamp", "mirror", "wrap"], segment.spline_endpoint
    )
    return segment


def _wizard_build_render_hints() -> dict[str, Any] | None:
    if not _prompt_yes_no("Configure advanced render hints (chroma/mp4)?", default=False):
        return None

    hints: dict[str, Any] = {}
    raw_chroma = _prompt_text("  chroma key color (R,G,B or #RRGGBB, blank for none)", "")
    if raw_chroma:
        _parse_rgb_color(raw_chroma)
        hints["chroma_key"] = raw_chroma
        hints["chroma_threshold"] = _prompt_float("  chroma threshold", 16.0, min_value=0.0)

    raw_bg = _prompt_text("  mp4 background color (R,G,B or #RRGGBB, blank for none)", "")
    if raw_bg:
        _parse_rgb_color(raw_bg)
        hints["mp4_background"] = raw_bg

    return hints if hints else None


def _wizard_output_target(default_output: str | None) -> Path:
    while True:
        mode = _prompt_choice("Output mode", ["new", "update"], "new")
        if mode == "new":
            path = Path(_prompt_text("Output file path", default_output or "timeline.wizard.json"))
            return path

        update_default = default_output or "timeline.example.json"
        path = Path(_prompt_text("Existing file to update", update_default))
        if path.exists():
            return path
        print(f"File does not exist: {path}")


def _build_timeline_data_for_wizard(
    input_dir: Path,
    keyframes: list[Path],
    base_segment: SegmentSpec,
    fps: int,
    render_hints: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[SegmentSpec]]:
    segments = [SegmentSpec(**vars(base_segment)) for _ in range(len(keyframes) - 1)]

    if _prompt_yes_no("Customize selected segments?", default=True):
        while True:
            raw = _prompt_text("Segment indices to edit (1-based, comma-separated)", "1")
            try:
                indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
            except ValueError:
                print("Please enter integers like 1,3")
                continue
            if not indices:
                print("Please choose at least one segment.")
                continue
            if any(i < 1 or i > len(segments) for i in indices):
                print("One or more segment indices are out of range.")
                continue
            for i in indices:
                segments[i - 1] = _wizard_edit_segment(segments[i - 1], i)
            break

    timeline_data: dict[str, Any] = {
        "keyframes": [str(path.relative_to(input_dir)) if not path.is_absolute() else str(path) for path in keyframes],
        "defaults": {
            "frames": base_segment.frames,
            "easing": base_segment.easing,
            "interpolation": base_segment.interpolation,
            "alpha_blend": base_segment.alpha_blend,
            "spline_tension": base_segment.spline_tension,
            "spline_bias": base_segment.spline_bias,
            "spline_continuity": base_segment.spline_continuity,
            "spline_endpoint": base_segment.spline_endpoint,
        },
        "segments": [_segment_to_json_dict(segment) for segment in segments],
    }
    if render_hints:
        timeline_data["render_hints"] = render_hints

    warnings, _, approx_gib, total_frames = _timeline_doctor_warnings(image_paths=keyframes, segments=segments, fps=fps)
    print(f"Planned frames: {total_frames}, estimated RGBA buffer: {approx_gib:.2f} GiB")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f" - {warning}")

    risky = any(
        [
            total_frames >= 2000,
            approx_gib >= 1.0,
        ]
    )
    if risky and not _prompt_yes_no("Risky settings detected. Continue writing timeline?", default=False):
        raise RuntimeError("Timeline wizard canceled by user.")

    return timeline_data, segments


def _run_timeline_wizard(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_path = _wizard_output_target(args.output)
    fps = _prompt_int("Target fps", args.fps, min_value=1)

    available = discover_images(input_dir)
    if len(available) < 2:
        raise ValueError(f"Need at least 2 PNG files in {input_dir}")

    keyframes = _wizard_pick_keyframes(available)
    preset_name, preset = _wizard_select_preset()
    print(f"Selected preset: {preset_name}")

    preset.frames = _prompt_int("Default frames per segment", preset.frames, min_value=1)
    if _prompt_yes_no("Adjust preset defaults before segment overrides?", default=False):
        preset = _wizard_edit_segment(preset, 0)

    render_hints = _wizard_build_render_hints()
    timeline_data, _ = _build_timeline_data_for_wizard(
        input_dir=input_dir,
        keyframes=keyframes,
        base_segment=preset,
        fps=fps,
        render_hints=render_hints,
    )

    if output_path.exists() and not _prompt_yes_no(f"Overwrite existing file {output_path}?", default=True):
        raise RuntimeError("Timeline wizard canceled by user.")

    output_path.write_text(json.dumps(timeline_data, indent=2), encoding="utf-8")

    schema_path = Path(args.schema)
    if schema_path.exists():
        schema_errors = _validate_timeline_schema(output_path, schema_path)
        if schema_errors:
            print("Warning: written timeline has schema issues:")
            for error in schema_errors:
                print(f" - {error}")

    print(f"Wrote timeline via wizard to {output_path}")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_keyframes_from_selector(input_dir: Path, selector: str | None) -> list[Path]:
    discovered = discover_images(input_dir)
    if selector is None or not selector.strip():
        if len(discovered) < 2:
            raise ValueError(f"Need at least 2 PNG files in {input_dir}")
        return discovered

    selected: list[Path] = []
    tokens = _split_csv(selector)
    if len(tokens) < 2:
        raise ValueError("keyframes selector must reference at least two items")

    for token in tokens:
        if token.isdigit():
            idx = int(token)
            if idx < 1 or idx > len(discovered):
                raise ValueError(f"Keyframe index out of range: {idx}")
            selected.append(discovered[idx - 1])
            continue

        path = _resolve_keyframe_path(token, input_dir)
        if not path.exists():
            raise ValueError(f"Keyframe path not found: {path}")
        selected.append(path)

    return selected


def _coerce_easing(value: Any, context: str) -> str:
    easing = str(value).strip().lower()
    allowed = {"linear", "ease-in", "ease-out", "ease-in-out", "smoothstep", "smootherstep"}
    if easing not in allowed:
        raise ValueError(f"Invalid easing in {context}: {easing}")
    return easing


def _coerce_interpolation(value: Any, context: str) -> str:
    mode = str(value).strip().lower()
    allowed = {"catmull-rom", "linear"}
    if mode not in allowed:
        raise ValueError(f"Invalid interpolation in {context}: {mode}")
    return mode


def _coerce_alpha_blend(value: Any, context: str) -> str:
    mode = str(value).strip().lower()
    allowed = {"premultiplied", "straight"}
    if mode not in allowed:
        raise ValueError(f"Invalid alpha_blend in {context}: {mode}")
    return mode


def _segment_with_overrides(base: SegmentSpec, raw: dict[str, Any], context: str, fps: int) -> SegmentSpec:
    duration_frames = raw.get("frames", raw.get("duration_frames", base.frames))
    duration_seconds = raw.get("duration_seconds")
    if duration_seconds is not None:
        duration_frames = max(1, int(round(float(duration_seconds) * fps)))

    try:
        frames = int(duration_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid duration in {context}") from exc
    if frames < 1:
        raise ValueError(f"frames in {context} must be >= 1")

    spline = _coerce_spline_control(
        tension_raw=raw.get("spline_tension", raw.get("tension", base.spline_tension)),
        bias_raw=raw.get("spline_bias", raw.get("bias", base.spline_bias)),
        continuity_raw=raw.get("spline_continuity", raw.get("continuity", base.spline_continuity)),
        endpoint_raw=raw.get("spline_endpoint", raw.get("endpoint", base.spline_endpoint)),
        context=context,
    )

    return SegmentSpec(
        frames=frames,
        easing=_coerce_easing(raw.get("easing", base.easing), context),
        interpolation=_coerce_interpolation(raw.get("interpolation", base.interpolation), context),
        alpha_blend=_coerce_alpha_blend(raw.get("alpha_blend", base.alpha_blend), context),
        spline_tension=spline.tension,
        spline_bias=spline.bias,
        spline_continuity=spline.continuity,
        spline_endpoint=spline.endpoint,
    )


def _apply_segment_overrides(segments: list[SegmentSpec], overrides_path: Path | None, fps: int) -> list[SegmentSpec]:
    if overrides_path is None:
        return segments
    payload = json.loads(overrides_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_overrides = payload.get("segments")
    else:
        raw_overrides = payload

    if not isinstance(raw_overrides, list):
        raise ValueError("segment-overrides file must be a list or object with 'segments' list")

    out = [SegmentSpec(**vars(segment)) for segment in segments]
    for raw in raw_overrides:
        if not isinstance(raw, dict):
            raise ValueError("Each segment override entry must be an object")
        raw_index = raw.get("index")
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Each segment override requires integer 'index'") from exc
        if index < 1 or index > len(out):
            raise ValueError(f"Segment override index out of range: {index}")

        context = f"segment override {index}"
        out[index - 1] = _segment_with_overrides(out[index - 1], raw, context=context, fps=fps)
    return out


def _run_timeline_compose(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    fps = int(args.fps)
    if fps < 1:
        raise ValueError("fps must be >= 1")

    keyframes = _resolve_keyframes_from_selector(input_dir, args.keyframes)

    preset_name = str(args.preset)
    if preset_name not in WIZARD_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    base = SegmentSpec(**vars(WIZARD_PRESETS[preset_name]))
    if args.frames_per_segment is not None:
        base.frames = int(args.frames_per_segment)
    base = _segment_with_overrides(
        base,
        {
            "easing": args.easing,
            "interpolation": args.interpolation,
            "alpha_blend": args.alpha_blend,
            "spline_tension": args.spline_tension,
            "spline_bias": args.spline_bias,
            "spline_continuity": args.spline_continuity,
            "spline_endpoint": args.spline_endpoint,
            "frames": base.frames,
        },
        context="compose defaults",
        fps=fps,
    )

    segments = [SegmentSpec(**vars(base)) for _ in range(len(keyframes) - 1)]
    overrides_path = Path(args.segment_overrides_file) if args.segment_overrides_file else None
    segments = _apply_segment_overrides(segments, overrides_path, fps=fps)

    warnings, _, approx_gib, total_frames = _timeline_doctor_warnings(keyframes, segments, fps=fps)
    risky = total_frames >= 2000 or approx_gib >= 1.0
    if risky and not args.allow_risky:
        raise RuntimeError(
            "Risky timeline (high frame count or memory estimate). Re-run with --allow-risky to continue."
        )

    render_hints: dict[str, Any] | None = None
    chroma = _chroma_key_from_args(args)
    bg = _parse_rgb_color(args.mp4_background) if args.mp4_background else None
    if chroma is not None or bg is not None:
        render_hints = {}
        if chroma is not None:
            render_hints["chroma_key"] = f"{chroma.color[0]},{chroma.color[1]},{chroma.color[2]}"
            render_hints["chroma_threshold"] = chroma.threshold
        if bg is not None:
            render_hints["mp4_background"] = f"{bg[0]},{bg[1]},{bg[2]}"

    timeline_data: dict[str, Any] = {
        "keyframes": [str(path.relative_to(input_dir)) if not path.is_absolute() else str(path) for path in keyframes],
        "defaults": {
            "frames": base.frames,
            "easing": base.easing,
            "interpolation": base.interpolation,
            "alpha_blend": base.alpha_blend,
            "spline_tension": base.spline_tension,
            "spline_bias": base.spline_bias,
            "spline_continuity": base.spline_continuity,
            "spline_endpoint": base.spline_endpoint,
        },
        "segments": [_segment_to_json_dict(segment) for segment in segments],
    }
    if render_hints:
        timeline_data["render_hints"] = render_hints

    if output_path.exists() and not args.overwrite:
        raise RuntimeError(f"Refusing to overwrite existing file: {output_path}. Use --overwrite to replace it.")

    output_path.write_text(json.dumps(timeline_data, indent=2), encoding="utf-8")

    schema_path = Path(args.schema)
    if schema_path.exists():
        schema_errors = _validate_timeline_schema(output_path, schema_path)
        if schema_errors:
            print(f"Wrote {output_path}, but schema validation failed ({len(schema_errors)} errors):")
            for error in schema_errors:
                print(f" - {error}")
            raise SystemExit(2)

    print(f"Wrote timeline via compose command to {output_path}")
    print(f" - Keyframes: {len(keyframes)}")
    print(f" - Segments: {len(segments)}")
    print(f" - Estimated frames: {total_frames}")
    print(f" - Estimated RGBA buffer: {approx_gib:.2f} GiB")
    if warnings:
        print(f" - Warnings: {len(warnings)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spline interpolation experiments for image sequences.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cmd_generate = subparsers.add_parser("generate-images", help="Generate canonical test images.")
    cmd_generate.add_argument("--output-dir", default="assets/test_images")
    cmd_generate.add_argument("--width", type=int, default=640)
    cmd_generate.add_argument("--height", type=int, default=640)
    cmd_generate.add_argument("--seed", type=int, default=7)

    cmd_list = subparsers.add_parser("list-images", help="List available keyframe images.")
    cmd_list.add_argument("--input-dir", default="assets/test_images")
    cmd_list.add_argument("--timeline", default=None, help="Optional timeline JSON to list ordered keyframes.")
    cmd_list.add_argument("--frames-per-segment", type=int, default=20)
    cmd_list.add_argument(
        "--easing",
        default="linear",
        choices=["linear", "ease-in", "ease-out", "ease-in-out", "smoothstep", "smootherstep"],
    )
    cmd_list.add_argument("--interpolation", default="catmull-rom", choices=["catmull-rom", "linear"])
    cmd_list.add_argument("--alpha-blend", default="premultiplied", choices=["straight", "premultiplied"])
    cmd_list.add_argument("--fps", type=int, default=24)
    _add_spline_control_args(cmd_list)

    cmd_render = subparsers.add_parser("render", help="Render a spline interpolation video/gif.")
    cmd_render.add_argument("--input-dir", default="assets/test_images")
    cmd_render.add_argument("--timeline", default=None, help="Optional timeline JSON for keyframe order and segments.")
    cmd_render.add_argument("--output", default="artifacts/spline_demo.gif")
    cmd_render.add_argument("--frames-per-segment", type=int, default=20)
    cmd_render.add_argument(
        "--easing",
        default="linear",
        choices=["linear", "ease-in", "ease-out", "ease-in-out", "smoothstep", "smootherstep"],
    )
    cmd_render.add_argument("--interpolation", default="catmull-rom", choices=["catmull-rom", "linear"])
    cmd_render.add_argument("--alpha-blend", default="premultiplied", choices=["straight", "premultiplied"])
    cmd_render.add_argument("--fps", type=int, default=24)
    _add_chroma_args(cmd_render)
    _add_mp4_output_args(cmd_render)
    _add_spline_control_args(cmd_render)

    cmd_export = subparsers.add_parser("export-frames", help="Export interpolated frames as PNG files.")
    cmd_export.add_argument("--input-dir", default="assets/test_images")
    cmd_export.add_argument("--timeline", default=None, help="Optional timeline JSON for keyframe order and segments.")
    cmd_export.add_argument("--output-dir", default="artifacts/frames")
    cmd_export.add_argument("--frames-per-segment", type=int, default=20)
    cmd_export.add_argument(
        "--easing",
        default="linear",
        choices=["linear", "ease-in", "ease-out", "ease-in-out", "smoothstep", "smootherstep"],
    )
    cmd_export.add_argument("--interpolation", default="catmull-rom", choices=["catmull-rom", "linear"])
    cmd_export.add_argument("--alpha-blend", default="premultiplied", choices=["straight", "premultiplied"])
    cmd_export.add_argument("--fps", type=int, default=24)
    _add_chroma_args(cmd_export)
    _add_spline_control_args(cmd_export)

    cmd_template = subparsers.add_parser("write-timeline-template", help="Write a starter JSON timeline file.")
    cmd_template.add_argument("--output", default="timeline.example.json")

    cmd_validate = subparsers.add_parser(
        "validate-timeline",
        help="Validate a timeline against JSON schema and runtime rules.",
    )
    cmd_validate.add_argument("--timeline", required=True, help="Path to timeline JSON file.")
    cmd_validate.add_argument("--schema", default="timeline.schema.json", help="Path to timeline JSON schema.")
    cmd_validate.add_argument("--input-dir", default="assets/test_images")
    _add_timeline_resolution_defaults(cmd_validate)

    cmd_doctor = subparsers.add_parser(
        "timeline-doctor",
        help="Show timeline diagnostics, resolved segment settings, and risk warnings.",
    )
    cmd_doctor.add_argument("--timeline", required=True, help="Path to timeline JSON file.")
    cmd_doctor.add_argument("--schema", default="timeline.schema.json", help="Path to timeline JSON schema.")
    cmd_doctor.add_argument("--input-dir", default="assets/test_images")
    _add_timeline_resolution_defaults(cmd_doctor)

    cmd_wizard = subparsers.add_parser(
        "timeline-wizard",
        help="Interactive wizard for creating or updating timeline JSON files.",
    )
    cmd_wizard.add_argument("--input-dir", default="assets/test_images")
    cmd_wizard.add_argument("--output", default=None, help="Optional default output path shown in prompts.")
    cmd_wizard.add_argument("--schema", default="timeline.schema.json")
    cmd_wizard.add_argument("--fps", type=int, default=24)

    cmd_compose = subparsers.add_parser(
        "timeline-compose",
        help="Non-interactive timeline creation command for scripts/automation.",
    )
    cmd_compose.add_argument("--input-dir", default="assets/test_images")
    cmd_compose.add_argument("--output", required=True)
    cmd_compose.add_argument("--schema", default="timeline.schema.json")
    cmd_compose.add_argument(
        "--keyframes",
        default=None,
        help="Comma-separated keyframe selectors. Use 1-based indices or filenames/paths.",
    )
    cmd_compose.add_argument(
        "--preset",
        default="cinematic-smooth",
        choices=sorted(WIZARD_PRESETS.keys()),
    )
    cmd_compose.add_argument("--frames-per-segment", type=int, default=None)
    cmd_compose.add_argument(
        "--easing",
        default="ease-in-out",
        choices=["linear", "ease-in", "ease-out", "ease-in-out", "smoothstep", "smootherstep"],
    )
    cmd_compose.add_argument("--interpolation", default="catmull-rom", choices=["catmull-rom", "linear"])
    cmd_compose.add_argument("--alpha-blend", default="premultiplied", choices=["straight", "premultiplied"])
    cmd_compose.add_argument("--fps", type=int, default=24)
    _add_spline_control_args(cmd_compose)
    _add_chroma_args(cmd_compose)
    _add_mp4_output_args(cmd_compose)
    cmd_compose.add_argument(
        "--segment-overrides-file",
        default=None,
        help="JSON file with per-segment overrides, entries containing 1-based 'index'.",
    )
    cmd_compose.add_argument("--overwrite", action="store_true")
    cmd_compose.add_argument("--allow-risky", action="store_true")

    return parser.parse_args()


def _timeline_template() -> dict[str, Any]:
    return {
        "keyframes": [
            "01_red_point_blue_stars.png",
            "02_blue_circle_indigo_to_blue_alpha_outside.png",
            "03_blue_point_rainbow_spiral.png",
            "04_darting_blue_line_rainbow_streaks.png",
        ],
        "defaults": {
            "frames": 20,
            "easing": "smoothstep",
            "interpolation": "catmull-rom",
            "alpha_blend": "premultiplied",
            "spline_tension": 0.1,
            "spline_bias": 0.0,
            "spline_continuity": -0.1,
            "spline_endpoint": "mirror",
        },
        "segments": [
            {
                "duration_frames": 12,
                "easing": "ease-in",
                "interpolation": "linear"
            },
            {
                "duration_frames": 24,
                "easing": "ease-in-out",
                "interpolation": "catmull-rom",
                "spline_tension": 0.25,
                "spline_continuity": 0.1
            },
            {
                "duration_seconds": 1.4,
                "easing": "ease-out",
                "alpha_blend": "straight",
                "spline_endpoint": "wrap"
            },
        ],
    }


def main() -> None:
    args = parse_args()

    if args.command == "generate-images":
        files = generate_test_images(
            output_dir=Path(args.output_dir),
            size=(args.width, args.height),
            seed=args.seed,
        )
        print(f"Generated {len(files)} images.")
        for file_path in files:
            print(f" - {file_path}")
        return

    if args.command == "list-images":
        timeline_path = Path(args.timeline) if args.timeline else None
        default_spec = _default_segment_spec_from_args(args)
        _, image_paths, segments = _load_animator_and_segments(
            input_dir=Path(args.input_dir),
            timeline_path=timeline_path,
            default_spec=default_spec,
            fps=args.fps,
        )
        print(f"Found {len(image_paths)} PNG files in {args.input_dir}")
        for path in image_paths:
            print(f" - {path}")
        print(f"Segments: {len(segments)}")
        for idx, segment in enumerate(segments):
            print(
                f"   segment {idx}: frames={segment.frames}, easing={segment.easing}, "
                f"interpolation={segment.interpolation}, alpha_blend={segment.alpha_blend}, "
                f"spline_tension={segment.spline_tension}, spline_bias={segment.spline_bias}, "
                f"spline_continuity={segment.spline_continuity}, spline_endpoint={segment.spline_endpoint}"
            )
        return

    if args.command == "render":
        timeline_path = Path(args.timeline) if args.timeline else None
        default_spec = _default_segment_spec_from_args(args)
        chroma_key = _chroma_key_from_args(args)
        mp4_background = _parse_rgb_color(args.mp4_background) if args.mp4_background else None
        if timeline_path is not None:
            hint_chroma, hint_bg = _load_render_hints_from_timeline(timeline_path)
            if chroma_key is None:
                chroma_key = hint_chroma
            if mp4_background is None:
                mp4_background = hint_bg
        animator, _, segments = _load_animator_and_segments(
            input_dir=Path(args.input_dir),
            timeline_path=timeline_path,
            default_spec=default_spec,
            fps=args.fps,
        )
        config = RenderConfig(fps=args.fps, mp4_background=mp4_background)
        output_path = Path(args.output)
        animator.render_video(output_path, config, segments, chroma_key=chroma_key)

        estimated_frames = sum(spec.frames for spec in segments) + 1
        print(f"Rendered {estimated_frames} frames to {output_path}")
        return

    if args.command == "export-frames":
        timeline_path = Path(args.timeline) if args.timeline else None
        default_spec = _default_segment_spec_from_args(args)
        chroma_key = _chroma_key_from_args(args)
        if timeline_path is not None and chroma_key is None:
            hint_chroma, _ = _load_render_hints_from_timeline(timeline_path)
            chroma_key = hint_chroma
        animator, _, segments = _load_animator_and_segments(
            input_dir=Path(args.input_dir),
            timeline_path=timeline_path,
            default_spec=default_spec,
            fps=args.fps,
        )
        written = animator.export_frames(Path(args.output_dir), segments, chroma_key=chroma_key)
        print(f"Wrote {len(written)} interpolated frames to {args.output_dir}")
        return

    if args.command == "write-timeline-template":
        output_path = Path(args.output)
        output_path.write_text(json.dumps(_timeline_template(), indent=2), encoding="utf-8")
        print(f"Wrote timeline template to {output_path}")
        return

    if args.command == "validate-timeline":
        timeline_path = Path(args.timeline)
        schema_path = Path(args.schema)
        schema_errors = _validate_timeline_schema(timeline_path=timeline_path, schema_path=schema_path)

        if schema_errors:
            print(f"Schema validation failed ({len(schema_errors)} errors):")
            for error in schema_errors:
                print(f" - {error}")
            raise SystemExit(2)

        default_spec = _default_segment_spec_from_args(args)
        image_paths, segments = _resolve_image_paths_and_segments(
            input_dir=Path(args.input_dir),
            timeline_path=timeline_path,
            default_spec=default_spec,
            fps=args.fps,
        )
        _load_render_hints_from_timeline(timeline_path)
        total_frames = sum(segment.frames for segment in segments) + 1
        print("Timeline validation passed.")
        print(f" - Keyframes: {len(image_paths)}")
        print(f" - Segments: {len(segments)}")
        print(f" - Estimated frames: {total_frames}")
        return

    if args.command == "timeline-doctor":
        timeline_path = Path(args.timeline)
        schema_path = Path(args.schema)
        schema_errors = _validate_timeline_schema(timeline_path=timeline_path, schema_path=schema_path)

        default_spec = _default_segment_spec_from_args(args)
        image_paths, segments = _resolve_image_paths_and_segments(
            input_dir=Path(args.input_dir),
            timeline_path=timeline_path,
            default_spec=default_spec,
            fps=args.fps,
        )

        warnings, max_size, approx_gib, total_frames = _timeline_doctor_warnings(
            image_paths=image_paths,
            segments=segments,
            fps=args.fps,
        )
        hint_chroma, hint_bg = _load_render_hints_from_timeline(timeline_path)

        print("Timeline doctor report")
        print(f" - keyframes: {len(image_paths)}")
        print(f" - segments: {len(segments)}")
        print(f" - estimated frames: {total_frames}")
        if args.fps > 0:
            print(f" - estimated duration: {total_frames / args.fps:.2f}s at {args.fps} fps")
        print(f" - largest keyframe: {max_size[0]}x{max_size[1]}")
        print(f" - estimated RGBA buffer: {approx_gib:.2f} GiB")
        print(f" - render hint chroma: {'set' if hint_chroma else 'none'}")
        print(f" - render hint mp4 background: {hint_bg if hint_bg else 'none'}")

        print("Resolved segments")
        for idx, segment in enumerate(segments):
            print(
                " - "
                f"{idx}: frames={segment.frames}, easing={segment.easing}, "
                f"interp={segment.interpolation}, alpha={segment.alpha_blend}, "
                f"tension={segment.spline_tension}, bias={segment.spline_bias}, "
                f"continuity={segment.spline_continuity}, endpoint={segment.spline_endpoint}"
            )

        if schema_errors:
            print(f"Schema issues ({len(schema_errors)}):")
            for error in schema_errors:
                print(f" - {error}")

        if warnings:
            print(f"Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f" - {warning}")
        else:
            print("Warnings (0): none")
        return

    if args.command == "timeline-wizard":
        try:
            _run_timeline_wizard(args)
        except RuntimeError as exc:
            print(str(exc))
            raise SystemExit(1) from exc
        return

    if args.command == "timeline-compose":
        try:
            _run_timeline_compose(args)
        except RuntimeError as exc:
            print(str(exc))
            raise SystemExit(1) from exc
        return

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
