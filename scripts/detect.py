
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from effort_wrapper import EffortDetector


def collect(folder, limit=None):
    files = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return files[:limit] if limit else files


def features_for(det, files):
    """Batch through the frozen backbone, skipping unreadable files."""
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
    return np.concatenate(feats), names


# ----------------------------------------------------------------------

def cmd_fit(args):
    det = EffortDetector(args.ckpt)

    real_files = collect(args.real, args.shots)
    fake_files = collect(args.fake, args.shots)
    if len(real_files) < args.shots or len(fake_files) < args.shots:
        raise SystemExit("not enough samples for the requested shot count")

    print(f"fitting on {len(real_files)} real + {len(fake_files)} fake")
    f_real, _ = features_for(det, real_files)
    f_fake, _ = features_for(det, fake_files)

    X = np.vstack([f_real, f_fake])
    y = np.r_[np.zeros(len(f_real)), np.ones(len(f_fake))].astype(int)

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(X), y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"scaler": scaler, "clf": clf,
                     "shots": args.shots,
                     "real_source": str(args.real),
                     "fake_source": str(args.fake),
                     "real_files": [p.name for p in real_files],
                     "fake_files": [p.name for p in fake_files]}, fh)

    print(f"saved head to {out}")
    print(f"  real support: {args.real}")
    print(f"  fake support: {args.fake}")
    print("  backbone unchanged - only this head was fitted")


def cmd_predict(args):
    det = EffortDetector(args.ckpt)
    with open(args.head, "rb") as fh:
        bundle = pickle.load(fh)
    scaler, clf = bundle["scaler"], bundle["clf"]

    files = collect(args.images)
    if not files:
        raise SystemExit(f"no images in {args.images}")

    feats, names = features_for(det, files)
    probs = clf.predict_proba(scaler.transform(feats))[:, 1]

    print(f"\nhead fitted on {bundle['shots']} samples/class "
          f"({Path(bundle['fake_source']).name} fakes)")
    print(f"threshold {args.threshold}\n")
    print(f"  {'P(AI-generated)':>16}  {'verdict':>12}   image")
    print("  " + "-" * 58)

    rows = []
    for n, p in sorted(zip(names, probs), key=lambda t: -t[1]):
        verdict = "AI-GENERATED" if p > args.threshold else "authentic"
        print(f"  {p:>16.4f}  {verdict:>12}   {n}")
        rows.append({"image": n, "p_fake": float(p), "verdict": verdict})

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"threshold": args.threshold,
                   "head": str(args.head),
                   "results": rows}, open(args.save, "w"), indent=2)
        print(f"\nsaved to {args.save}")

    n_fake = sum(r["verdict"] == "AI-GENERATED" for r in rows)
    print(f"\n{n_fake}/{len(rows)} flagged as AI-generated")


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("ckpt")
    f.add_argument("--real", required=True)
    f.add_argument("--fake", required=True)
    f.add_argument("--shots", type=int, default=5)
    f.add_argument("--out", default="../results/head.pkl")
    f.set_defaults(func=cmd_fit)

    p = sub.add_parser("predict")
    p.add_argument("ckpt")
    p.add_argument("--head", default="../results/head.pkl")
    p.add_argument("--images", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save", default=None)
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()