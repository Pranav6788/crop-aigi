# scripts/test_compression.py
import sys, numpy as np
from pathlib import Path
from PIL import Image
from effort_wrapper import EffortDetector, fake_prob

src = Path(sys.argv[2]); out = src.parent / "test_compressed"
out.mkdir(exist_ok=True)

for p in sorted(src.iterdir()):
    if p.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
    img = Image.open(p).convert("RGB")
    w, h = img.size; s = 512 / max(w, h)
    if s < 1: img = img.resize((round(w*s), round(h*s)), Image.LANCZOS)
    img.save(out / p.name, quality=75)          # ImageNet-like

np.set_printoptions(precision=2, suppress=True)
det = EffortDetector(sys.argv[1])
for label, folder in [("ORIGINAL", src), ("COMPRESSED 512px q75", out)]:
    names, _, logits = det.score_folder(folder)
    p = fake_prob(logits)
    y = np.array([0 if n.startswith('real') else 1 for n in names])
    from sklearn.metrics import roc_auc_score
    print(f"\n{label}:  AUROC {roc_auc_score(y,p):.3f}  "
          f"real {p[y==0].mean():.3f}  fake {p[y==1].mean():.3f}")
    for n, lg, pr in sorted(zip(names, logits, p), key=lambda t:-t[2]):
        print(f"  {pr:.4f}  logits {lg}  {n}")