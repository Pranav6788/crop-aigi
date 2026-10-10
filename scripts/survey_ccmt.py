# scripts/survey_ccmt.py
from pathlib import Path
from PIL import Image
from collections import Counter
import sys

root = Path(sys.argv[1])
for cls in sorted(p for p in root.iterdir() if p.is_dir()):
    sizes, files = Counter(), list(cls.glob("*"))
    for f in files:
        try: sizes[Image.open(f).size] += 1
        except: pass
    print(f"\n{cls.name}  ({len(files)} images)")
    for s, n in sizes.most_common(5):
        print(f"   {str(s):18s} {n:5d}  ({n/len(files)*100:.0f}%)")