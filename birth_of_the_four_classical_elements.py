from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


RGBA = tuple[int, int, int, int]


def _rainbow_rgb(t: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(max(0.0, min(1.0, t)), 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _base_space(size: tuple[int, int], color: RGBA = (4, 6, 20, 255)) -> Image.Image:
    return Image.new("RGBA", size, color)


def red_point_with_blue_stars(size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    image = _base_space(size, (4, 4, 18, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = size

    star_count = max(140, (w * h) // 2500)
    for _ in range(star_count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        radius = int(rng.integers(1, 3))
        intensity = int(rng.integers(150, 255))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(30, 90, intensity, 255))

    cx, cy = w // 2, h // 2
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(255, 20, 20, 255))
    return image


def blue_circle_indigo_to_blue(size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    del rng
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    radius = min(w, h) * 0.36

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    t = np.clip(dist / radius, 0.0, 1.0)

    indigo = np.array([75.0, 0.0, 130.0])
    blue = np.array([25.0, 120.0, 255.0])
    rgb = (indigo * (1.0 - t[..., None]) + blue * t[..., None]).astype(np.uint8)

    arr = np.zeros((h, w, 4), dtype=np.uint8)
    mask = dist <= radius
    arr[..., :3] = rgb
    arr[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def blue_point_and_rainbow_spiral(size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    del rng
    image = _base_space(size, (3, 3, 10, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = size
    cx, cy = w / 2.0, h / 2.0

    turns = 9
    points = 3600
    max_radius = min(w, h) * 0.46

    for i in range(points):
        t = i / (points - 1)
        angle = turns * 2.0 * math.pi * t
        radius = max_radius * t
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        cr, cg, cb = _rainbow_rgb(t)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(cr, cg, cb, 235))

    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(20, 120, 255, 255))
    return image


def darting_blue_line_and_rainbow_star_trails(size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    w, h = size
    image = _base_space(size, (2, 2, 18, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    trail_count = max(160, (w * h) // 2000)
    for _ in range(trail_count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(max(5, w // 50), max(12, w // 18)))
        tilt = float(rng.uniform(-0.3, 0.3))
        x2 = x + int(length * math.cos(tilt))
        y2 = y + int(length * math.sin(tilt))
        cr, cg, cb = _rainbow_rgb(float(rng.random()))
        draw.line((x, y, x2, y2), fill=(cr, cg, cb, 190), width=2)

    y_mid = int(h * 0.45)
    draw.line((int(w * 0.08), y_mid + int(h * 0.12), int(w * 0.92), y_mid - int(h * 0.12)), fill=(20, 100, 255, 255), width=10)
    return image


def violet_triangle_blue_fill(size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    del rng
    w, h = size
    image = Image.new("RGBA", size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    p1 = (w // 2, int(h * 0.16))
    p2 = (int(w * 0.18), int(h * 0.84))
    p3 = (int(w * 0.82), int(h * 0.84))

    draw.polygon((p1, p2, p3), fill=(15, 90, 240, 255))
    draw.line((p1, p2), fill=(148, 0, 211, 255), width=6)
    draw.line((p2, p3), fill=(148, 0, 211, 255), width=6)
    draw.line((p3, p1), fill=(148, 0, 211, 255), width=6)
    return image


def square_multicolor_edges(size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    del rng
    w, h = size
    image = Image.new("RGBA", size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    margin = int(min(w, h) * 0.2)
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin

    # The prompt repeats "left edge" twice; this sketch uses green on the right edge.
    draw.line((x0, y0, x1, y0), fill=(150, 90, 40, 255), width=8)    # upper edge brown
    draw.line((x0, y0, x0, y1), fill=(40, 100, 255, 255), width=8)    # left edge blue
    draw.line((x0, y1, x1, y1), fill=(220, 40, 40, 255), width=8)     # lower edge red
    draw.line((x1, y0, x1, y1), fill=(30, 170, 60, 255), width=8)     # right edge green
    return image


def generate_test_images(output_dir: Path, size: tuple[int, int], seed: int = 7) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    generators = [
        ("red_point_blue_stars", red_point_with_blue_stars),
        ("blue_circle_indigo_to_blue_alpha_outside", blue_circle_indigo_to_blue),
        ("blue_point_rainbow_spiral", blue_point_and_rainbow_spiral),
        ("darting_blue_line_rainbow_streaks", darting_blue_line_and_rainbow_star_trails),
        ("violet_triangle_blue_fill", violet_triangle_blue_fill),
        ("square_multicolor_edges", square_multicolor_edges),
    ]

    written: list[Path] = []
    manifest: list[dict[str, str]] = []
    for idx, (name, fn) in enumerate(generators, start=1):
        filename = f"{idx:02d}_{name}.png"
        target = output_dir / filename
        image = fn(size, rng)
        image.save(target)
        written.append(target)
        manifest.append({"file": filename, "generator": fn.__name__})

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test images for spline animation experiments.",
    )
    parser.add_argument("--output-dir", default="assets/test_images", help="Directory for generated test images.")
    parser.add_argument("--width", type=int, default=640, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=640, help="Image height in pixels.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible stars/streaks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    files = generate_test_images(output_dir=output_dir, size=(args.width, args.height), seed=args.seed)

    print(f"Wrote {len(files)} images to {output_dir}")
    for path in files:
        print(f" - {path}")


if __name__ == "__main__":
    main()
