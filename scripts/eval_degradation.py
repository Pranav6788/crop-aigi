
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from effort_wrapper import EffortDetector, fake_prob

CONDITIONS = ["T0", "T1", "T2", "T3"]


def classify_pool(stem: str, fake_markers):
    """Return 'fake' or 'real' from the filename, or None to skip."""
    for m in fake_markers:
        if m in stem:
            return "fake"
    return "real"


def load_condition(det, degraded: Path, cond: str, cache: Path,
                   fake_markers, real_markers):
    """Extract (or load) features for every image at one condition."""
    cache.mkdir(parents=True, exist_ok=True)
    fx, ly, fn = (cache / f"{cond}_{s}" for s in
                  ("feats.npy", "labels.npy", "names.json"))

    if fx.exists():
        print(f"  {cond}: cached")
        return np.load(fx), np.load(ly), json.load(open(fn))

    files = sorted(p for p in degraded.glob(f"*_{cond}.jpg"))
    keep, labels = [], []
    for p in files:
        stem = p.stem[: -len(f"_{cond}")]
        if any(m in stem for m in fake_markers):
            keep.append(p); labels.append(1)
        elif any(m in stem for m in real_markers):
            keep.append(p); labels.append(0)
        # anything else (e.g. P2) is excluded from this comparison

    if not keep:
        raise SystemExit(f"no images matched at {cond}")

    print(f"  {cond}: extracting {len(keep)}", end="", flush=True)
    feats, names, ok_labels = [], [], []
    for i in range(0, len(keep), 16):
        chunk = keep[i:i + 16]
        chunk_lab = labels[i:i + 16]
        imgs, ok, lab = [], [], []
        for p, l in zip(chunk, chunk_lab):
            try:
                im = Image.open(p); im.load()
                imgs.append(im); ok.append(p); lab.append(l)
            except Exception as e:
                print(f"\n    skip {p.name}: {e}")
        if not imgs:
            continue
        f, _ = det.score_batch(imgs)
        feats.append(f)
        names += [p.name for p in ok]
        ok_labels += lab
    print()

    X = np.concatenate(feats)
    y = np.array(ok_labels)
    np.save(fx, X); np.save(ly, y); json.dump(names, open(fn, "w"))
    return X, y, names


def zero_shot(det, degraded, cond, fake_markers, real_markers, cache):
    """Effort's released head at this condition."""
    lg_cache = cache / f"{cond}_logits.npy"
    if lg_cache.exists():
        d = np.load(lg_cache, allow_pickle=True).item()
        return d["scores"], d["labels"]

    files = sorted(p for p in degraded.glob(f"*_{cond}.jpg"))
    keep, labels = [], []
    for p in files:
        stem = p.stem[: -len(f"_{cond}")]
        if any(m in stem for m in fake_markers):
            keep.append(p); labels.append(1)
        elif any(m in stem for m in real_markers):
            keep.append(p); labels.append(0)

    scores, ok_lab = [], []
    for i in range(0, len(keep), 16):
        chunk, lab = keep[i:i + 16], labels[i:i + 16]
        imgs, l2 = [], []
        for p, l in zip(chunk, lab):
            try:
                im = Image.open(p); im.load(); imgs.append(im); l2.append(l)
            except Exception:
                pass
        if not imgs:
            continue
        _, logits = det.score_batch(imgs)
        scores.append(fake_prob(logits)); ok_lab += l2

    s = np.concatenate(scores); y = np.array(ok_lab)
    np.save(lg_cache, {"scores": s, "labels": y})
    return s, y


def few_shot(X, y, k, trials, seed):
    idx_r, idx_f = np.where(y == 0)[0], np.where(y == 1)[0]
    if k > min(len(idx_r), len(idx_f)) - 3:
        return None
    rng = np.random.default_rng(seed)
    aucs, baccs = [], []
    for _ in range(trials):
        sup = np.r_[rng.choice(idx_r, k, replace=False),
                    rng.choice(idx_f, k, replace=False)]
        test = np.setdiff1d(np.arange(len(y)), sup)
        sc = StandardScaler().fit(X[sup])
        clf = LinearSVC(C=1.0, max_iter=5000).fit(sc.transform(X[sup]), y[sup])
        s = clf.decision_function(sc.transform(X[test]))
        aucs.append(roc_auc_score(y[test], s))
        baccs.append(balanced_accuracy_score(y[test], (s > 0).astype(int)))
    a, b = np.array(aucs), np.array(baccs)
    return dict(auroc=a.mean(), auroc_ci=1.96 * a.std(ddof=1) / np.sqrt(len(a)),
                bacc=b.mean(), bacc_ci=1.96 * b.std(ddof=1) / np.sqrt(len(b)),
                n_test=len(test))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--degraded", required=True)
    ap.add_argument("--cache", default="../data/features_degraded")
    ap.add_argument("--fake-markers", nargs="+", default=["_P3"],
                    help="substrings identifying fake images")
    ap.add_argument("--real-markers", nargs="+", default=["healthy"],
                    help="substrings identifying real images")
    ap.add_argument("--shots", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # a real marker that also matches a fake marker would mislabel
    # everything, so resolve fakes first (done in load_condition)
    det = EffortDetector(args.ckpt)
    degraded, cache = Path(args.degraded), Path(args.cache)

    print("\nfeatures")
    data = {}
    for c in CONDITIONS:
        X, y, _ = load_condition(det, degraded, c, cache,
                                 args.fake_markers, args.real_markers)
        s, ys = zero_shot(det, degraded, c, args.fake_markers,
                          args.real_markers, cache)
        data[c] = (X, y, s, ys)
        print(f"       {c}: {len(y)} images  ({(y==0).sum()} real, "
              f"{(y==1).sum()} fake)")

    results = {}

    print("\nZERO-SHOT (released head)")
    print(f"  {'cond':>4}  {'AUROC':>7}  {'bAcc':>7}  {'TPR':>7}  {'TNR':>7}")
    for c in CONDITIONS:
        _, _, s, y = data[c]
        pred = (s > 0.5).astype(int)
        r = dict(auroc=roc_auc_score(y, s),
                 bacc=balanced_accuracy_score(y, pred),
                 tpr=pred[y == 1].mean(), tnr=1 - pred[y == 0].mean())
        results.setdefault(c, {})["zero_shot"] = {k: float(v) for k, v in r.items()}
        print(f"  {c:>4}  {r['auroc']:7.3f}  {r['bacc']:7.3f}  "
              f"{r['tpr']:7.3f}  {r['tnr']:7.3f}")

    for k in args.shots:
        print(f"\nFEW-SHOT k={k}")
        print(f"  {'cond':>4}  {'AUROC':>16}  {'bAcc':>16}   n_test")
        for c in CONDITIONS:
            X, y, _, _ = data[c]
            r = few_shot(X, y, k, args.trials, args.seed)
            if r is None:
                print(f"  {c:>4}  skipped - not enough samples")
                continue
            results.setdefault(c, {})[f"{k}_shot"] = {
                kk: float(vv) for kk, vv in r.items()}
            print(f"  {c:>4}  {r['auroc']:.3f} +/- {r['auroc_ci']:.3f}   "
                  f"{r['bacc']:.3f} +/- {r['bacc_ci']:.3f}   {r['n_test']}")

    # ---- axis decomposition and threshold gap ----
    print("\nAXIS DECOMPOSITION (relative to T0)")
    for tag in ["zero_shot"] + [f"{k}_shot" for k in args.shots]:
        if tag not in results["T0"]:
            continue
        base_a = results["T0"][tag]["auroc"]
        base_b = results["T0"][tag]["bacc"]
        print(f"\n  {tag}")
        print(f"    {'cond':>4}  {'dAUROC':>8}  {'dbAcc':>8}  {'gap':>8}   axis")
        labels = {"T1": "capture", "T2": "channel", "T3": "combined"}
        for c in ["T1", "T2", "T3"]:
            if tag not in results.get(c, {}):
                continue
            da = results[c][tag]["auroc"] - base_a
            db = results[c][tag]["bacc"] - base_b
            gap = db - da          # negative: accuracy fell further than ranking
            results[c][tag]["d_auroc"] = float(da)
            results[c][tag]["d_bacc"] = float(db)
            results[c][tag]["threshold_gap"] = float(gap)
            print(f"    {c:>4}  {da:+8.3f}  {db:+8.3f}  {gap:+8.3f}   {labels[c]}")

        if all(tag in results.get(c, {}) for c in ["T1", "T2", "T3"]):
            additive = results["T1"][tag]["d_auroc"] + results["T2"][tag]["d_auroc"]
            actual = results["T3"][tag]["d_auroc"]
            print(f"    T1+T2 predicts {additive:+.3f}, T3 actual {actual:+.3f} "
                  f"-> interaction {actual - additive:+.3f}")

    out = Path("../results"); out.mkdir(exist_ok=True)
    json.dump(results, open(out / "degradation_results.json", "w"), indent=2)
    print(f"\nsaved {out/'degradation_results.json'}")


if __name__ == "__main__":
    main()