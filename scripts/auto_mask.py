
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png"}


def leaf_region(bgr):
    """
    Leaf tissue: green-ish, reasonably bright, AND in focus.

    Hue alone is insufficient for wider-framed images, where blurred
    background vegetation passes any colour test. Local variance of the
    Laplacian separates the sharp foreground leaf from defocused
    background.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colour = (h > 20) & (h < 95) & (s > 30) & (v > 45)

    # local sharpness
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sharp = cv2.GaussianBlur(lap * lap, (31, 31), 0)
    thr = np.quantile(sharp, 0.55)          # keep the sharper half
    focused = sharp > thr

    ok = (colour & focused).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ok = cv2.morphologyEx(ok, cv2.MORPH_CLOSE, k)
    ok = cv2.morphologyEx(ok, cv2.MORPH_OPEN, k)

    # keep only the largest connected region - the main leaf
    n, lab, stats, _ = cv2.connectedComponentsWithStats((ok > 0).astype(np.uint8), 8)
    if n > 1:
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        ok = np.where(lab == biggest, 255, 0).astype(np.uint8)

    if ok.mean() < 40:
        ok[:] = 255
    return ok


def organic_mask(h, w, target_area, scale, rng, allowed=None):
    """
    Threshold blurred noise to get irregular blobs.

    scale controls blob size: larger blur -> fewer, larger regions.
    The threshold is chosen by percentile so the realised area lands on
    target regardless of the noise draw.
    """
    noise = rng.random((h, w)).astype(np.float32)
    k = int(scale) | 1                       # kernel must be odd
    field = cv2.GaussianBlur(noise, (k, k), scale / 3.0)

    if allowed is not None:
        valid = allowed > 0
        if valid.sum() == 0:
            valid = np.ones_like(field, bool)
        # aim for target_area of the FRAME, taken from allowed pixels only
        want = target_area * h * w
        frac_of_valid = min(0.95, want / max(1, valid.sum()))
        thr = np.quantile(field[valid], 1.0 - frac_of_valid)
        mask = ((field >= thr) & valid).astype(np.uint8) * 255
    else:
        thr = np.quantile(field, 1.0 - target_area)
        mask = (field >= thr).astype(np.uint8) * 255

    # tidy speckle without rounding the outlines off
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    return mask


def drop_tiny(mask, min_frac=0.015):
    """Remove specks too small for the model to render as anything."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    floor = min_frac * mask.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= floor:
            out[lab == i] = 255
    return out


def make_mask(bgr, rng, args):
    h, w = bgr.shape[:2]
    allowed = leaf_region(bgr) if not args.no_leaf_filter else None

    # sample a target area from the band so the set has a real spread,
    # rather than every mask sitting at the same size
    for _ in range(12):
        target = rng.uniform(args.min_area, args.max_area)
        scale = rng.uniform(args.scale * 0.7, args.scale * 1.3)
        m = organic_mask(h, w, target, scale, rng, allowed)
        m = drop_tiny(m)
        frac = (m > 0).mean()
        if args.min_area * 0.85 <= frac <= args.max_area * 1.15:
            return m, frac
    return m, frac                            # best effort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--min-area", type=float, default=0.15)
    ap.add_argument("--max-area", type=float, default=0.35)
    ap.add_argument("--scale", type=float, default=28,
                    help="blur kernel; larger -> fewer, bigger blobs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-leaf-filter", action="store_true")
    ap.add_argument("--preview", action="store_true",
                    help="also write overlay images for visual checking")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    prev = dst / "_preview"
    if args.preview:
        prev.mkdir(exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no images in {src}")

    rng = np.random.default_rng(args.seed)
    log_path = Path(args.log) if args.log else dst.parent / f"{dst.name}_log.csv"
    fh = open(log_path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["filename", "area_fraction", "width", "height", "seed"])

    fracs = []
    for p in files:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"  skip unreadable {p.name}")
            continue

        m, frac = make_mask(bgr, rng, args)
        cv2.imwrite(str(dst / f"{p.stem}_mask.png"), m)
        wr.writerow([p.name, f"{frac:.4f}", bgr.shape[1], bgr.shape[0], args.seed])
        fracs.append(frac)

        if args.preview:
            ov = bgr.copy()
            ov[m > 0] = (0, 0, 255)
            cv2.imwrite(str(prev / f"{p.stem}_overlay.jpg"),
                        cv2.addWeighted(bgr, 0.6, ov, 0.4, 0))

    fh.close()
    a = np.array(fracs)
    print(f"wrote {len(a)} masks to {dst}")
    print(f"area: mean {a.mean()*100:.1f}%  min {a.min()*100:.1f}%  "
          f"max {a.max()*100:.1f}%  std {a.std()*100:.1f}%")
    print(f"log: {log_path}")
    if args.preview:
        print(f"overlays: {prev}")


if __name__ == "__main__":
    main()