"""
Mask drawing tool for benchmark synthesis.

Paint white regions where AI-generated damage should be inpainted.
Everything outside the mask is preserved bit-identically by the
inpainting model, which is what gives real/fake semantic pairing.

Advantages over GIMP: masks are saved automatically with a consistent
naming convention, and mask area fraction is logged per image, which
you need for the report.

Controls
    left-drag    paint mask
    right-drag   erase
    [ / ]        smaller / larger brush
    r            reset current mask
    s            save and go to next image
    n            skip without saving
    q            quit

Usage
    pip install opencv-python
    python mask_tool.py --in data/P1 --out data/masks --log data/mask_log.csv

Note: needs a local display. Run on your laptop, not in Colab.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

TARGET_MIN = 0.15   # mask area band from the report
TARGET_MAX = 0.35
DISPLAY_MAX = 900   # window size cap


class Painter:
    def __init__(self, img):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.mask = np.zeros((self.h, self.w), np.uint8)
        self.brush = max(12, self.w // 40)
        self.drawing = False
        self.erasing = False

        self.scale = min(1.0, DISPLAY_MAX / max(self.h, self.w))

    def on_mouse(self, event, x, y, flags, _):
        gx, gy = int(x / self.scale), int(y / self.scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = self.erasing = False

        if self.drawing:
            cv2.circle(self.mask, (gx, gy), self.brush, 255, -1)
        elif self.erasing:
            cv2.circle(self.mask, (gx, gy), self.brush, 0, -1)

    def area_fraction(self):
        return float((self.mask > 0).sum()) / (self.h * self.w)

    def render(self):
        """Image with the mask tinted, plus a status bar."""
        overlay = self.img.copy()
        overlay[self.mask > 0] = (0, 0, 255)
        view = cv2.addWeighted(self.img, 0.6, overlay, 0.4, 0)

        if self.scale < 1.0:
            view = cv2.resize(view, None, fx=self.scale, fy=self.scale)

        frac = self.area_fraction()
        ok = TARGET_MIN <= frac <= TARGET_MAX
        colour = (0, 200, 0) if ok else (0, 0, 255)
        txt = f"area {frac*100:5.1f}%  target {TARGET_MIN*100:.0f}-{TARGET_MAX*100:.0f}%  brush {self.brush}"
        cv2.rectangle(view, (0, 0), (view.shape[1], 26), (30, 30, 30), -1)
        cv2.putText(view, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--log", default="mask_log.csv")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    # resume support: skip anything already masked
    done = {p.stem.replace("_mask", "") for p in dst.glob("*_mask.png")}
    todo = [p for p in files if p.stem not in done]

    print(f"{len(files)} images, {len(done)} already masked, {len(todo)} remaining")

    log_path = Path(args.log)
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["filename", "area_fraction", "width", "height"])

    cv2.namedWindow("mask")

    for i, path in enumerate(todo):
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip unreadable {path.name}")
            continue

        p = Painter(img)
        cv2.setMouseCallback("mask", p.on_mouse)
        print(f"[{i+1}/{len(todo)}] {path.name}")

        while True:
            cv2.imshow("mask", p.render())
            k = cv2.waitKey(20) & 0xFF

            if k == ord("["):
                p.brush = max(3, p.brush - 4)
            elif k == ord("]"):
                p.brush += 4
            elif k == ord("r"):
                p.mask[:] = 0
            elif k == ord("n"):
                break
            elif k == ord("q"):
                log.close(); cv2.destroyAllWindows(); return
            elif k == ord("s"):
                frac = p.area_fraction()
                if frac == 0:
                    print("  empty mask - not saved")
                    continue
                if not (TARGET_MIN <= frac <= TARGET_MAX):
                    print(f"  WARNING area {frac*100:.1f}% outside target band")
                out = dst / f"{path.stem}_mask.png"
                cv2.imwrite(str(out), p.mask)
                writer.writerow([path.name, f"{frac:.4f}", p.w, p.h])
                log.flush()
                print(f"  saved {out.name}  area {frac*100:.1f}%")
                break

    log.close()
    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    main()