from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from birth_of_the_four_classical_elements import generate_test_images


@dataclass
class RenderConfig:
    frames_per_segment: int = 20
    fps: int = 24


class SplineAnimator:
    """Minimal Catmull-Rom image spline interpolator."""

    def __init__(self, keyframes: list[np.ndarray]):
        if len(keyframes) < 2:
            raise ValueError("Need at least two keyframes.")
        self.keyframes = [frame.astype(np.float32) for frame in keyframes]

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

    def interpolated_frames(self, frames_per_segment: int) -> list[np.ndarray]:
        if frames_per_segment < 1:
            raise ValueError("frames_per_segment must be >= 1")

        frames: list[np.ndarray] = []
        n = len(self.keyframes)

        for i in range(n - 1):
            p0 = self.keyframes[max(0, i - 1)]
            p1 = self.keyframes[i]
            p2 = self.keyframes[i + 1]
            p3 = self.keyframes[min(n - 1, i + 2)]

            for step in range(frames_per_segment):
                t = step / frames_per_segment
                frame = self._catmull_rom(p0, p1, p2, p3, t)
                frames.append(np.clip(frame, 0, 255).astype(np.uint8))

        frames.append(np.clip(self.keyframes[-1], 0, 255).astype(np.uint8))
        return frames

    def export_frames(self, output_dir: Path, frames_per_segment: int) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.interpolated_frames(frames_per_segment)

        written: list[Path] = []
        for idx, frame in enumerate(frames):
            path = output_dir / f"frame_{idx:04d}.png"
            Image.fromarray(frame, mode="RGBA").save(path)
            written.append(path)
        return written

    def render_video(self, output_path: Path, config: RenderConfig) -> None:
        frames = self.interpolated_frames(config.frames_per_segment)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            iio.imwrite(output_path, frames, format="GIF", duration=1.0 / config.fps, loop=0)
            return

        if suffix == ".mp4":
            rgb_frames = [frame[..., :3] for frame in frames]
            iio.imwrite(output_path, rgb_frames, fps=config.fps, codec="libx264")
            return

        raise ValueError("Unsupported output format. Use .gif or .mp4")


def discover_images(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.glob("*.png") if p.is_file()])


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

    cmd_render = subparsers.add_parser("render", help="Render a spline interpolation video/gif.")
    cmd_render.add_argument("--input-dir", default="assets/test_images")
    cmd_render.add_argument("--output", default="artifacts/spline_demo.gif")
    cmd_render.add_argument("--frames-per-segment", type=int, default=20)
    cmd_render.add_argument("--fps", type=int, default=24)

    cmd_export = subparsers.add_parser("export-frames", help="Export interpolated frames as PNG files.")
    cmd_export.add_argument("--input-dir", default="assets/test_images")
    cmd_export.add_argument("--output-dir", default="artifacts/frames")
    cmd_export.add_argument("--frames-per-segment", type=int, default=20)

    return parser.parse_args()


def _load_animator(input_dir: Path) -> tuple[SplineAnimator, list[Path]]:
    image_paths = discover_images(input_dir)
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 PNG files in {input_dir}")
    return SplineAnimator.from_paths(image_paths), image_paths


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
        image_paths = discover_images(Path(args.input_dir))
        print(f"Found {len(image_paths)} PNG files in {args.input_dir}")
        for path in image_paths:
            print(f" - {path}")
        return

    if args.command == "render":
        animator, image_paths = _load_animator(Path(args.input_dir))
        config = RenderConfig(frames_per_segment=args.frames_per_segment, fps=args.fps)
        output_path = Path(args.output)
        animator.render_video(output_path, config)

        estimated_frames = (len(image_paths) - 1) * args.frames_per_segment + 1
        print(f"Rendered {estimated_frames} frames to {output_path}")
        return

    if args.command == "export-frames":
        animator, _ = _load_animator(Path(args.input_dir))
        written = animator.export_frames(Path(args.output_dir), args.frames_per_segment)
        print(f"Wrote {len(written)} interpolated frames to {args.output_dir}")
        return

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
