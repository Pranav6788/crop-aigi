# scripts/build_pools.py
from pathlib import Path
from PIL import Image
import shutil, sys

MAIZE = Path(sys.argv[1])          # .../Maize
N = 200

def build(src_name, dst_name, n):
    src = MAIZE / src_name
    dst = Path("../data") / dst_name
    dst.mkdir(parents=True, exist_ok=True)

    kept = 0
    for f in sorted(src.glob("*")):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            if Image.open(f).size != (400, 400):
                continue
        except Exception:
            continue
        shutil.copy(f, dst / f.name.replace(" ", "_"))
        kept += 1
        if kept >= n:
            break
    print(f"{dst_name}: {kept} images from {src_name}")

build("healthy",     "P1", N)
build("leaf blight", "P2", N)