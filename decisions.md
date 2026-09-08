## 2026-09-08 — Preprocessing
Resize MUST use cv2.resize INTER_LINEAR, matching demo.py line 173.
PIL BILINEAR anti-aliases on downscale and destroys the signal:
AUROC 0.640 (PIL) vs 0.960 (cv2) on the same 10-image check.
Evidence that Effort depends on low-level resampling artifacts,
not only semantics — relevant to Ch2 §11.1.