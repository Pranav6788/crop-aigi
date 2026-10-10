
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.linalg import det

from effort_wrapper import EffortDetector, fake_prob


# ----------------------------------------------------------------------

def get_features(det, folder, cache_dir, tag):
    """Extract once, cache to disk. Everything downstream is then CPU-cheap."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fcache = cache_dir / f"{tag}_feats.npy"
    lcache = cache_dir / f"{tag}_logits.npy"
    ncache = cache_dir / f"{tag}_names.json"

    if fcache.exists():
        print(f"  {tag}: loaded from cache")
        return (np.load(fcache), np.load(lcache),
                json.load(open(ncache)))

    print(f"  {tag}: extracting")
    names, feats, logits = det.score_folder(folder)
    np.save(fcache, feats)
    np.save(lcache, logits)
    json.dump(names, open(ncache, "w"))
    return feats, logits, names


def evaluate(scores, labels, threshold=0.5):
    pred = (scores > threshold).astype(int)
    tpr = pred[labels == 1].mean() if (labels == 1).any() else float("nan")
    tnr = 1 - pred[labels == 0].mean() if (labels == 0).any() else float("nan")
    return dict(
        auroc=roc_auc_score(labels, scores),
        bacc=balanced_accuracy_score(labels, pred),
        tpr=tpr,
        tnr=tnr,
    )


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--real", required=True, help="P2 - real damaged")
    ap.add_argument("--fake", required=True, help="P3 - synthetic")
    ap.add_argument("--cache", default="../data/features")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--trials", type=int, default=20,
                    help="random support draws per shot count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    det = EffortDetector(args.ckpt)
    cache = Path(args.cache)

    f_real, lg_real, _ = get_features(det, args.real, cache, Path(args.real).name)
    f_fake, lg_fake, _ = get_features(det, args.fake, cache, Path(args.fake).name)

    X = np.vstack([f_real, f_fake])
    y = np.r_[np.zeros(len(f_real)), np.ones(len(f_fake))].astype(int)
    zs = np.r_[fake_prob(lg_real), fake_prob(lg_fake)]

    print(f"\nreal n={len(f_real)}   fake n={len(f_fake)}   dim={X.shape[1]}")

    # ---- zero-shot baseline: Effort's own head, no adaptation ----
    base = evaluate(zs, y)
    print("\nZERO-SHOT (released head, no adaptation)")
    print(f"  AUROC {base['auroc']:.3f}   bAcc {base['bacc']:.3f}   "
          f"TPR {base['tpr']:.3f}   TNR {base['tnr']:.3f}")

    # ---- few-shot: freeze backbone, refit head ----
    print("\nFEW-SHOT (frozen backbone, SVM head refit)")
    print(f"  {'shots':>6}  {'AUROC':>16}  {'bAcc':>16}   n_test")

    idx_real = np.where(y == 0)[0]
    idx_fake = np.where(y == 1)[0]
    results = {}

    for k in args.shots:
        if k > min(len(idx_real), len(idx_fake)) - 3:
            print(f"  {k:>6}  skipped - not enough samples")
            continue

        aucs, baccs = [], []
        rng = np.random.default_rng(args.seed)

        for _ in range(args.trials):
            sup = np.r_[rng.choice(idx_real, k, replace=False),
                        rng.choice(idx_fake, k, replace=False)]
            test = np.setdiff1d(np.arange(len(y)), sup)

            scaler = StandardScaler().fit(X[sup])          # fit on support only
            clf = LinearSVC(C=1.0, max_iter=5000)
            clf.fit(scaler.transform(X[sup]), y[sup])

            s = clf.decision_function(scaler.transform(X[test]))
            aucs.append(roc_auc_score(y[test], s))
            baccs.append(balanced_accuracy_score(y[test], (s > 0).astype(int)))

        a, b = np.array(aucs), np.array(baccs)
        ci_a = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
        ci_b = 1.96 * b.std(ddof=1) / np.sqrt(len(b))
        print(f"  {k:>6}  {a.mean():.3f} +/- {ci_a:.3f}   "
              f"{b.mean():.3f} +/- {ci_b:.3f}   {len(test)}")
        results[k] = dict(auroc=float(a.mean()), auroc_ci=float(ci_a),
                          bacc=float(b.mean()), bacc_ci=float(ci_b))

    out = Path("../results"); out.mkdir(exist_ok=True)
    json.dump({"zero_shot": base, "few_shot": results,
               "trials": args.trials,
               "n_real": len(f_real), "n_fake": len(f_fake)},
              open(out / "fewshot_results.json", "w"), indent=2)
    print(f"\nsaved to {out/'fewshot_results.json'}")


if __name__ == "__main__":
    main()