
import argparse
import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CAPTURE = {
    "B0": None,                                                   # identity
    "B1": dict(long_edge=1024, noise=2.0,  blur=0.5, jpeg=85),    # mild
    "B2": dict(long_edge=640,  noise=10.0, blur=1.0, jpeg=65),    # severe
}

CHANNEL_LONG_EDGE = 1600   #messaging apps resize toward this
CHANNEL_JPEG_Q    = 75     #approximate re-encode quality per cycle

SCREENSHOT_LONG_EDGE = 1080   #device screen
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
    w, h = img.size
    longest = max(w, h)
    if longest <= target:
        return img
    scale = target / longest
    arr = cv2.resize(np.asarray(img), (round(w * scale), round(h * scale)),
                     interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(arr)


def add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
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
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------
# Axes
# ----------------------------------------------------------------------

def apply_capture(img: Image.Image, level: str) -> Image.Image:
    p = CAPTURE[level]
    if p is None:
        return img
    img = resize_long_edge(img, p["long_edge"])
    img = add_gaussian_noise(img, p["noise"])
    img = gaussian_blur(img, p["blur"])
    img = jpeg_roundtrip(img, p["jpeg"])
    return img


def apply_channel(img: Image.Image, cycles: int) -> Image.Image:
    for _ in range(cycles):
        img = resize_long_edge(img, CHANNEL_LONG_EDGE)
        img = jpeg_roundtrip(img, CHANNEL_JPEG_Q)
    return img


def apply_screenshot(img: Image.Image) -> Image.Image:
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