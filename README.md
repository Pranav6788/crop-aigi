# Calibrated, Domain-Robust Detection of AI-Generated Imagery
### for Crop Insurance Evidence Verification

BCSE497J Project-I · VIT Chennai · Fall 2026–27

A benchmark and evaluation of foundation-model AI-generated image
detection on agricultural crop-damage imagery, with lightweight
in-domain adaptation.

## Findings

| | AUROC | TPR | TNR |
|---|---|---|---|
| Zero-shot (Effort released head) | 0.700 | 0.050 | 0.990 |
| Adapted classifier (test, n=121) | 0.999 | 0.925 | 0.988 |

Zero-shot detection catches 5% of manipulated images. The failure is in
the decision layer, not the representation — refitting a 1,025-parameter
linear head on in-domain samples recovers it.

Under degradation, capture and channel effects are superadditive:
measured separately they predict a 0.051 AUROC loss; applied together
they cost 0.099.

## Benchmark

| Pool | n | Content |
|---|---|---|
| `data/P1` | 198 | Real healthy maize (CCMT) |
| `data/P2` | 200 | Real pest-damaged maize (CCMT) |
| `data/P3` | 198 | SD 1.5 inpainted from P1 |
| `data/P3_chatgpt` | 20 | ChatGPT-edited from P1 |

All 400×400. Synthetic images are localised inpaints — everything
outside the mask is bit-identical to the source, so real and fake
differ only in the manipulated region.

`data/masks_p1_auto/` holds the masks used; `data/P3/generation_log.json`
records prompt, guidance, steps and per-image seeds.

`data/P3_chatgpt_raw/` is the original 1024×1024 output. Generated
interactively and not seed-reproducible, so retained as the only source.

## Reproducing

Requires the CCMT raw set (Mendeley 10.17632/bwh3zbpkpv.1, CC BY 4.0,
not redistributed here) and the Effort checkpoint
`effort_clip_L14_trainOn_sdv14.pth` from the authors' release.

```bash
pip install -r requirements.txt

# degradation conditions
python scripts/degrade.py --in data/base --out data/degraded --seed 0

# few-shot sweep
python scripts/fewshot.py <ckpt> --real data/real_v2 --fake data/fake_v2 \
    --shots 1 5 10 20 50 100

# degradation evaluation
python scripts/eval_degradation.py <ckpt> --degraded data/degraded \
    --fake-markers _P3 --real-markers healthy fall_armyworm leaf_beetle

# train and run the classifier
python scripts/train_classifier.py <ckpt> --real data/P1 data/P2 \
    --fake data/P3 data/P3_chatgpt --out results/classifier_v2.pkl
python scripts/detect.py predict <ckpt> --head results/classifier_v2.pkl \
    --images <folder>
```

## Scope

Validated on CCMT maize imagery captured with a single camera.
Generalisation to other capture sources is in progress. Few-shot
adaptation is generator-specific — a head fitted on one generator does
not transfer to another.

## Attribution

CCMT dataset: Mendeley Data, DOI 10.17632/bwh3zbpkpv.1, CC BY 4.0.
Effort: Yan et al., ICML 2025.
