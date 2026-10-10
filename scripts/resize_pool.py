

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def resize_square(img: Image.Image, size: int, crop: bool) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]

    if crop:
        # centre-crop to square first, then resize - preserves aspect ratio
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        arr = arr[top:top + side, left:left + side]

    interp = cv2.INTER_AREA if max(arr.shape[:2]) > size else cv2.INTER_LINEAR
    arr = cv2.resize(arr, (size, size), interpolation=interp)
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--no-crop", action="store_true",
                    help="squash to square instead of centre-cropping")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images found in {src}")

    n, sizes = 0, set()
    for p in files:
        try:
            img = Image.open(p)
            img.load()
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue

        sizes.add(img.size)
        out = resize_square(img, args.size, crop=not args.no_crop)
        out.save(dst / f"{p.stem}.jpg", quality=args.quality)
        n += 1

    print(f"wrote {n} images to {dst} at {args.size}x{args.size}")
    print(f"source sizes seen: {sorted(sizes)}")
    if not args.no_crop:
        print("mode: centre-crop to square, then resize")
    else:
        print("mode: squash to square (aspect ratio not preserved)")


if __name__ == "__main__":
    main()