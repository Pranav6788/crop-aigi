import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler

from effort_wrapper import EffortDetector

EXTS = {".jpg", ".jpeg", ".png"}


def source_stem(name: str) -> str:
    s = Path(name).stem
    s = re.sub(r"^(real_|fake_|synth_)", "", s)
    s = re.sub(r"_P[0-9][a-z]*.*$", "", s)
    return s


def extract(det, folder, label):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)
    feats, names = [], []
    for i in range(0, len(files), 16):
        chunk, imgs, ok = files[i:i + 16], [], []
        for p in chunk:
            try:
                im = Image.open(p); im.load()
                imgs.append(im); ok.append(p)
            except Exception as e:
                print(f"  skip {p.name}: {e}")
        if not imgs:
            continue
        f, _ = det.score_batch(imgs)
        feats.append(f)
        names += [p.name for p in ok]
    X = np.concatenate(feats)
    print(f"  {folder}: {len(names)} images, label={label}")
    return X, names, np.full(len(names), label)


def split_by_source(names, rng, frac=(0.6, 0.2, 0.2)):
    sources = sorted({source_stem(n) for n in names})
    rng.shuffle(sources)
    n = len(sources)
    n_tr, n_va = int(n * frac[0]), int(n * frac[1])
    assign = {}
    for i, s in enumerate(sources):
        assign[s] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return np.array([assign[source_stem(n)] for n in names]), len(sources)


def report(name, y, s, thr):
    pred = (s > thr).astype(int)
    tpr = pred[y == 1].mean() if (y == 1).any() else float("nan")
    tnr = 1 - pred[y == 0].mean() if (y == 0).any() else float("nan")
    auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    bacc = balanced_accuracy_score(y, pred)
    print(f"  {name:6s} n={len(y):4d}  AUROC {auc:.3f}  bAcc {bacc:.3f}  "
          f"TPR {tpr:.3f}  TNR {tnr:.3f}")
    return dict(n=int(len(y)), auroc=float(auc), bacc=float(bacc),
                tpr=float(tpr), tnr=float(tnr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    # ap.add_argument("--real", required=True)
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--fake", nargs="+", required=True)
    ap.add_argument("--out", default="../results/classifier.pkl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=0.10,
                    help="threshold chosen at this FPR on validation")
    args = ap.parse_args()

    det = EffortDetector(args.ckpt)
    print("\nextracting features")

    Xs, names, ys, srcs = [], [], [], []
    for folder in args.real:
        X, n, y = extract(det, folder, 0)
        Xs.append(X); names += n; ys.append(y); srcs += ["real"] * len(n)
    for folder in args.fake:
        X, n, y = extract(det, folder, 1)
        Xs.append(X); names += n; ys.append(y); srcs += [Path(folder).name] * len(n)

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    names = np.array(names)
    srcs = np.array(srcs)

    rng = np.random.default_rng(args.seed)
    split, n_sources = split_by_source(names, rng)
    print(f"\n{len(y)} images from {n_sources} source images")
    for s in ("train", "val", "test"):
        m = split == s
        print(f"  {s:6s} {m.sum():4d}  ({(y[m]==0).sum()} real, {(y[m]==1).sum()} fake)")

    tr, va, te = split == "train", split == "val", split == "test"
    if (y[va] == 1).sum() == 0 or (y[te] == 1).sum() == 0:
        raise SystemExit("val or test has no fakes - need more synthetic images")

    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = (scaler.transform(X[m]) for m in (tr, va, te))

    print("\nselecting C on validation")
    best = (None, -1)
    for C in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        clf = LogisticRegression(C=C, max_iter=5000,
                                 class_weight="balanced").fit(Xtr, y[tr])
        auc = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
        print(f"  C={C:<8} val AUROC {auc:.3f}")
        if auc > best[1]:
            best = (C, auc)
    C = best[0]
    print(f"  chosen C={C}")

    clf = LogisticRegression(C=C, max_iter=5000,
                             class_weight="balanced").fit(Xtr, y[tr])

    s_va = clf.predict_proba(Xva)[:, 1]
    fpr, tpr_c, thrs = roc_curve(y[va], s_va)
    idx = np.argmin(np.abs(fpr - args.target_fpr))
    thr = float(thrs[idx])
    print(f"\nthreshold {thr:.4f} (val FPR {fpr[idx]:.3f}, TPR {tpr_c[idx]:.3f})")

    print("\nresults")
    res = {}
    for nm, m, Xm in (("train", tr, Xtr), ("val", va, Xva), ("test", te, Xte)):
        res[nm] = report(nm, y[m], clf.predict_proba(Xm)[:, 1], thr)

    print("\nper-generator (test)")
    s_te = clf.predict_proba(Xte)[:, 1]
    for g in sorted(set(srcs[te])):
        gm = srcs[te] == g
        if gm.sum() == 0:
            continue
        caught = (s_te[gm] > thr).sum() if g != "real" else None
        if g == "real":
            print(f"  {g:14s} n={gm.sum():3d}  "
                  f"correctly passed {(s_te[gm] <= thr).sum()}/{gm.sum()}")
        else:
            print(f"  {g:14s} n={gm.sum():3d}  caught {caught}/{gm.sum()}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"scaler": scaler, "clf": clf, "threshold": thr,
                     "C": C, "shots": "full",
                     "real_source": args.real,
                     "fake_source": ", ".join(args.fake),
                     "n_train": int(tr.sum()), "metrics": res}, fh)
    json.dump(res, open(out.with_suffix(".json"), "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()