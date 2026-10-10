## 2026-09-08 — Preprocessing
Resize MUST use cv2.resize INTER_LINEAR, matching demo.py line 173.
PIL BILINEAR anti-aliases on downscale and destroys the signal:
AUROC 0.640 (PIL) vs 0.960 (cv2) on the same 10-image check.
Evidence that Effort depends on low-level resampling artifacts,
not only semantics — relevant to Ch2 §11.1.

## 2026-09-08 — CCMT raw is 400x400 ROI crops
Uniform 400x400 across all maize classes (99-100%). Eliminates
resolution as a confound.
Images are lesion ROI crops, not field-scale photographs. Maps to
the "zoom-level damage photograph" half of PMFBY capture protocols,
not the "corner shot" half. Field-scale evaluation = future work.

## 2026-09-08 — Zero-shot baseline on real CCMT maize (n=20 each)
healthy:     max 0.0324, median ~0.0018
leaf blight: max 0.0224, median ~0.0011
Real agricultural images score confidently real. No systematic
difference between healthy and diseased — lesion presence does not
bias the detector, so P2-vs-P3 discrimination is not confounded
by damage itself.

## 2026-09-08 — SD 2.1 inpainting unavailable
stabilityai/stable-diffusion-2-inpainting returns 401 (repo removed
or gated; multiple reports late 2025 for stabilityai SD 2.x repos).
Substituted <whichever works>. B-Free used SD 2.1; the relevant
discipline is localised inpainting with semantic pairing, not the
specific checkpoint. Model availability is itself a reproducibility
hazard for benchmark construction — worth a line in Ch 4 limitations.

## 2026-09-09 — Damage class switched to pest damage
SD 1.5 inpainting could not produce convincing blight lesions:
at guidance 7.5-20 it regenerated healthy tissue or added discrete
surface objects rather than in-plane necrosis. Diffusion inpainting
favours structural/object insertion over subtle surface-texture
modification. Switched to pest feeding damage (holes, ragged edges),
which matches that capability profile.
Constrains the realistic threat model: a fraudster using consumer
SD 1.5 tools faces the same limitation.

## 2026-09-09 — Model comparison at matched settings (g=14, 75 steps)
SD 1.5 inpainting: 2/4 convincing feeding holes
SDXL inpainting:   0/4 holes; produced in-plane brown streaking instead
SDXL is trained at 1024x1024; at 400x400 it upscales then downscales,
degrading fine structural detail. SD 1.5 selected.
Note the inverse capability: SDXL rendered subtle surface discolouration
(closer to blight) better than SD 1.5, while SD 1.5 rendered structural
openings better. Generative capability is damage-type dependent — a
constraint on any threat model assuming a single accessible tool.

## 2026-09-09 — Content confound identified
P1 vs P2 (both real, differ only in damage): 5-shot AUROC 0.998
P2 vs P3: 0.991  |  P1 vs P3 (source-matched): 0.849

The P2/P3 comparison is confounded: a head separates real healthy
from real damaged at 0.998, so the 0.991 largely reflects damage
content rather than manipulation. P1 vs P3 controls source and
capture session; 0.849 is the defensible manipulation-detection
figure. Residual confound: P3 has damage, P1 does not.

## 2026-09-09 — Cross-generator transfer fails
Test set: 5 SD-inpainted fakes + 5 real, held out.
  head fitted on SD fakes (10-shot):      3/5 caught
  head fitted on ChatGPT fakes (10-shot): 0/5 caught
Same backbone, same test images, same shot count.
Few-shot adaptation appears to learn generator-specific artifacts
rather than manipulation in general. Consistent with Corvi et al.
cross-generator findings, observed here at the adaptation layer.
Caveat: n=5 test, single support draw. Needs averaging over draws.
Untested: reverse direction (SD-fitted -> ChatGPT fakes).

## 2026-09-09 — Out-of-corpus false positives confirmed
Classifier: in-domain 5/5 fakes caught, 5/5 reals passed.
Novel ChatGPT fake (unseen): 0.9927, correctly flagged.
Real web photograph: 0.9460 — FALSE POSITIVE.
Second such case (earlier real.jpg scored 0.93).

Head fitted on 198 CCMT reals, all Canon EOS Rebel T7, all 400x400,
uniform JPEG history. Real images outside that distribution fall on
the fake side. Empirical confirmation of the limitation stated in
Ch1 §1.4.4. Mitigation: diversify the real class across sources.

## Generator selection
Compared N inpainting checkpoints on a fixed 5-image pilot with
identical masks, prompt, guidance and steps. Selected X on visual
comparison against real damaged imagery. Note that photorealism
fine-tunes sit further from the detector's training distribution
(SD 1.4) than base SD 1.5, which is representative of the threat
model: an adversary uses whatever produces convincing output.

## 2026-09-12 — Damage class constrained by generator capability
Tested across four attempts: blight (failed), leaf spot (failed,
both SD 1.5 and Realistic Vision), holes (works), whole-image
regeneration (works).

Diffusion inpainting inserts structure but does not modify existing
surface texture. Confirmed across base SD 1.5, SDXL, DreamShaper 8
and Realistic Vision 5.1 at matched settings.

Consequence: P4 (damage-matched pairing) is not achievable with
accessible tools. Reporting P1 vs P3 with the P1 vs P2 content
baseline stated alongside, so the confound is quantified rather
than hidden.

This is itself a constraint on the threat model — a fraudster using
consumer tools faces the same limitation, and can convincingly
fabricate structural damage but not subtle disease symptoms.

## 2026-09-12 — Content baseline vs manipulation detection (full pools, n≈198 each)

              5-shot   10-shot   20-shot   50-shot   100-shot
P1 vs P2      0.998    0.999     1.000     1.000     1.000
P1 vs P3      0.895    0.934     0.972     0.993     0.999

Content (real healthy vs real damaged) separates at 0.998 with five
samples. Manipulation detection needs ~100 samples to match it, so
manipulation is the harder task in this feature space and the P3 pool
is not trivially separable.

Zero-shot: P1/P2 AUROC 0.616, P1/P3 AUROC 0.748 — the released head is
more sensitive to manipulation than to damage content, but TPR 0.051
means it cannot act on that sensitivity.

Curve does not saturate by 10 shots, contrary to FSD's reported
behaviour on web-scale benchmarks. This domain needs more adaptation
data than the literature predicts.

## 2026-09-12 — Degradation results (P1 vs P3+P3c, n=416 per condition)

Zero-shot TPR: T0 0.041, T1 0.000, T2 0.000, T3 0.000
Zero-shot AUROC: 0.766 / 0.605 / 0.711 / 0.528

20-shot AUROC: 0.968 / 0.926 / 0.950 / 0.838

Axes are superadditive. T1+T2 predicts -0.061 at 20-shot; T3 measures
-0.130. Interaction of -0.070, consistent at 5, 10 and 20 shots.
Compound capture-and-channel shift costs roughly double the sum of
its parts. No published work measures this combination.

Capture degradation costs more than channel degradation at every
shot count (-0.043 vs -0.018 at 20-shot).

Zero-shot threshold gap reaches +0.220 at T3: AUROC falls 0.238 while
balanced accuracy falls 0.018, because the head already predicts
"real" almost everywhere. Strongest evidence for H2.

## Outstanding — validation control
Initial control set used web-sourced "real" images of unverifiable
provenance; result discarded. Redo with ImageNet or COCO validation
images and SD 1.5 generations to confirm the wrapper detects fakes in
Effort's training domain.


### data/P3_chatgpt_raw/
Original ChatGPT image-edit outputs at 1024×1024, before resizing.
Generated interactively, so not reproducible from a seed — retained
as the only source. `scripts/resize_pool.py` produces data/P3_chatgpt/
at 400×400 to match the benchmark resolution.