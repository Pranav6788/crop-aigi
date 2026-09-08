"""
Degradation pipeline for the agricultural AIGI benchmark.

Three independently controllable axes:
  B - capture conditions (low-cost sensor simulation)
  C - transmission channel (messaging-app recompression cycles)
  D - screenshot resubmission

Four evaluation conditions are produced:
  T0 = B0 C0 no-screenshot   (clean reference)
  T1 = B1 C0 no-screenshot   (capture axis isolated)
  T2 = B0 C1 no-screenshot   (channel axis isolated)
  T3 = B2 C3 screenshot      (realistic worst case)

Usage:
    python degrade.py --in data/base --out data/degraded

Input directory should contain images named like:
    img0042_P1.jpg / img0042_P2.jpg / img0042_P3.jpg
Output is written as:
    img0042_P3_T2.jpg
"""

import argparse
import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------
# Parameter table.  Tune these after the pilot, then FREEZE them and
# record the frozen values in the report.
# ----------------------------------------------------------------------

CAPTURE = {
    "B0": None,                                                   # identity
    "B1": dict(long_edge=1024, noise=2.0,  blur=0.5, jpeg=85),    # mild
    "B2": dict(long_edge=640,  noise=10.0, blur=1.0, jpeg=65),    # severe
}

CHANNEL_LONG_EDGE = 1600   # messaging apps resize toward this
CHANNEL_JPEG_Q    = 75     # approximate re-encode quality per cycle

SCREENSHOT_LONG_EDGE = 1080   # device screen
SCREENSHOT_JPEG_Q    = 80

CONDITIONS = {
    "T0": dict(capture="B0", cycles=0, screenshot=False),
    "T1": dict(capture="B1", cycles=0, screenshot=False),
    "T2": dict(capture="B0", cycles=1, screenshot=False),
    "T3": dict(capture="B2", cycles=3, screenshot=True),
}


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------

def resize_long_edge(img: Image.Image, target: int) -> Image.Image:
    """
    Resize so the longest side equals `target`. Never upscales.

    Uses cv2 INTER_LINEAR, NOT PIL. Two reasons:
      1. It matches the detector's own preprocessing (demo.py line 173).
         PIL's BILINEAR anti-aliases on downscale and removes exactly the
         high-frequency content the detector reads. Measured on a 10-image
         reproduction check: AUROC 0.640 with PIL vs 0.960 with cv2.
      2. Phone cameras and messaging apps do not anti-alias either, so
         cv2 is the more faithful model of the real channel.
    """
    w, h = img.size
    longest = max(w, h)
    if longest <= target:
        return img
    scale = target / longest
    arr = cv2.resize(np.asarray(img), (round(w * scale), round(h * scale)),
                     interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(arr)


def add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Additive Gaussian noise on an 8-bit scale."""
    if sigma <= 0:
        return img
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    """Encode to JPEG in memory and decode back. Strips all metadata."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------
# Axes
# ----------------------------------------------------------------------

def apply_capture(img: Image.Image, level: str) -> Image.Image:
    """
    Fixed operation order approximating a physical imaging pipeline:
    resolution -> sensor noise -> optical blur -> in-camera JPEG.
    The order matters; do not shuffle it between runs.
    """
    p = CAPTURE[level]
    if p is None:
        return img
    img = resize_long_edge(img, p["long_edge"])
    img = add_gaussian_noise(img, p["noise"])
    img = gaussian_blur(img, p["blur"])
    img = jpeg_roundtrip(img, p["jpeg"])
    return img


def apply_channel(img: Image.Image, cycles: int) -> Image.Image:
    """One cycle = one send/receive hop through a messaging app."""
    for _ in range(cycles):
        img = resize_long_edge(img, CHANNEL_LONG_EDGE)
        img = jpeg_roundtrip(img, CHANNEL_JPEG_Q)
    return img


def apply_screenshot(img: Image.Image) -> Image.Image:
    """Resample to a screen, then re-encode. Discards all metadata."""
    img = resize_long_edge(img, SCREENSHOT_LONG_EDGE)
    return jpeg_roundtrip(img, SCREENSHOT_JPEG_Q)


def apply_condition(img: Image.Image, cond: str) -> Image.Image:
    c = CONDITIONS[cond]
    img = apply_capture(img, c["capture"])
    img = apply_channel(img, c["cycles"])
    if c["screenshot"]:
        img = apply_screenshot(img)
    return img


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixes the noise draw; report this in the paper")
    args = ap.parse_args()

    np.random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not files:
        raise SystemExit(f"no images found in {src}")

    n = 0
    for path in files:
        img = Image.open(path).convert("RGB")
        stem = path.stem                      # e.g. img0042_P3
        for cond in CONDITIONS:
            out = apply_condition(img.copy(), cond)
            out.save(dst / f"{stem}_{cond}.jpg", quality=95)
            n += 1

    print(f"wrote {n} images ({len(files)} sources x {len(CONDITIONS)} conditions)")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()