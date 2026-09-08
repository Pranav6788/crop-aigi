"""
Minimal Effort (ICML 2025) inference wrapper.

Loads the released Effort checkpoint into a stock HuggingFace
CLIPVisionModel, bypassing the DeepfakeBench framework entirely.

Why this works
--------------
The checkpoint stores CLIP ViT-L/14 attention projections in
SVD-decomposed form:

    weight_main   (1024, 1024)   frozen principal components
    S_residual    (1,)           singular value  of the rank-1 update
    U_residual    (1024, 1)      left  vector
    V_residual    (1, 1024)      right vector

At inference the decomposition is irrelevant - only the effective
weight matters:

    W_eff = weight_main + U_residual @ diag(S_residual) @ V_residual

So we reconstruct W_eff once and load it as an ordinary Linear
weight. No custom modules, no framework, no Python 3.7.

Outputs both the 1024-d pooled feature (for few-shot head fitting)
and the 2-class logits from the released head.

Usage
-----
    from effort_wrapper import EffortDetector

    det = EffortDetector("weights/effort_genimage.pth")
    feats, logits = det.score_folder("data/degraded")
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionConfig, CLIPVisionModel
import cv2


from device import get_device

# ----------------------------------------------------------------------
# Preprocessing constants - taken from
# Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort.yaml
# These are CLIP's normalisation values, NOT ImageNet's. Do not change.
# ----------------------------------------------------------------------
RESOLUTION = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

# CLIP ViT-L/14 vision tower. 257 position embeddings = (224/14)^2 + 1.
VIT_L14 = dict(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_channels=3,
    image_size=224,
    patch_size=14,
    hidden_act="quick_gelu",       # CLIP uses quick_gelu, not gelu
    layer_norm_eps=1e-5,
)


def merge_svd(state):
    """
    Collapse SVD-decomposed projections into ordinary weight matrices.

    Input keys look like:
        module.backbone.encoder.layers.0.self_attn.k_proj.weight_main
        module.backbone.encoder.layers.0.self_attn.k_proj.S_residual
        module.backbone.encoder.layers.0.self_attn.k_proj.U_residual
        module.backbone.encoder.layers.0.self_attn.k_proj.V_residual

    Output:
        vision_model.encoder.layers.0.self_attn.k_proj.weight
    """
    # strip DataParallel prefix, split backbone from head
    backbone, head = {}, {}
    for k, v in state.items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k.startswith("backbone."):
            backbone[k[len("backbone."):]] = v
        elif k.startswith("head."):
            head[k[len("head."):]] = v

    # group by the parameter's parent module
    groups = {}
    for k, v in backbone.items():
        parent, _, leaf = k.rpartition(".")
        groups.setdefault(parent, {})[leaf] = v

    merged, n_merged = {}, 0
    for parent, params in groups.items():
        if "weight_main" in params:
            W = params["weight_main"]
            U = params["U_residual"]          # (out, r)
            S = params["S_residual"]          # (r,)
            V = params["V_residual"]          # (r, in)
            W_eff = W + (U * S) @ V           # rank-r update, r == 1 here
            merged[f"vision_model.{parent}.weight"] = W_eff
            if "bias" in params:
                merged[f"vision_model.{parent}.bias"] = params["bias"]
            n_merged += 1
        else:
            for leaf, v in params.items():
                key = f"vision_model.{parent}.{leaf}" if parent else f"vision_model.{leaf}"
                merged[key] = v

    return merged, head, n_merged


class EffortDetector:
    def __init__(self, ckpt_path, device=None, resize_mode="resize"):
        """
        resize_mode:
          "resize"      - squash to 224x224 (default)
          "center_crop" - resize short side to 224, then centre crop

        Check demo.py in the official repo and match whichever they use.
        Getting this wrong shifts scores without raising any error.
        """
        self.device = device or get_device()
        self.resize_mode = resize_mode

        # map_location="cpu" is required: the checkpoint was saved on a
        # CUDA machine and loading CUDA tensors straight to MPS fails.
        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw.get("state_dict", raw)

        backbone_sd, head_sd, n = merge_svd(state)
        print(f"merged {n} SVD projections")

        cfg = CLIPVisionConfig(**VIT_L14)
        self.model = CLIPVisionModel(cfg)

        missing, unexpected = self.model.load_state_dict(backbone_sd, strict=False)
        # position_ids is a non-persistent buffer in recent transformers
        missing = [k for k in missing if "position_ids" not in k]
        if missing:
            print(f"WARNING missing {len(missing)} keys, e.g. {missing[:5]}")
        if unexpected:
            print(f"WARNING unexpected {len(unexpected)} keys, e.g. {unexpected[:5]}")
        if not missing and not unexpected:
            print("backbone loaded cleanly")

        self.head = nn.Linear(1024, 2)
        self.head.load_state_dict({"weight": head_sd["weight"],
                                   "bias": head_sd["bias"]})

        self.model.eval().to(self.device)
        self.head.eval().to(self.device)

        self._mean = torch.tensor(MEAN).view(3, 1, 1)
        self._std = torch.tensor(STD).view(3, 1, 1)

    # ------------------------------------------------------------------

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        # Must match demo.py exactly: cv2.resize INTER_LINEAR, not PIL.
        # PIL's BILINEAR anti-aliases on downscale; cv2's does not.
        # For a forensic detector the difference is large.
        arr = np.asarray(img.convert("RGB"))
        arr = cv2.resize(arr, (RESOLUTION, RESOLUTION),
                        interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(arr.copy()).float().div(255.0).permute(2, 0, 1)
        return (x - self._mean) / self._std

    @torch.no_grad()
    def score_batch(self, images):
        """images: list of PIL.Image -> (features [N,1024], logits [N,2])"""
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        feats = self.model(pixel_values=x).pooler_output   # post_layernorm(CLS)
        logits = self.head(feats)
        return feats.cpu().numpy(), logits.cpu().numpy()

    def score_folder(self, folder, batch_size=16, pattern="*"):
        """Returns (filenames, features [N,1024], logits [N,2])."""
        files = sorted(p for p in Path(folder).glob(pattern)
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise SystemExit(f"no images in {folder} matching {pattern}")

        names, F, L = [], [], []
        for i in range(0, len(files), batch_size):
            chunk = files[i:i + batch_size]
            imgs = [Image.open(p) for p in chunk]
            f, l = self.score_batch(imgs)
            names += [p.name for p in chunk]
            F.append(f)
            L.append(l)
            print(f"  {min(i + batch_size, len(files))}/{len(files)}", end="\r")
        print()
        return names, np.concatenate(F), np.concatenate(L)


def fake_prob(logits):
    """P(fake). Assumes class 1 == fake - VERIFY this empirically."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit("usage: python effort_wrapper.py <ckpt.pth> <image_folder>")

    det = EffortDetector(sys.argv[1])
    names, feats, logits = det.score_folder(sys.argv[2])
    p = fake_prob(logits)

    print(f"\nfeatures {feats.shape}   logits {logits.shape}\n")
    for n, s in sorted(zip(names, p), key=lambda t: -t[1]):
        print(f"  {s:.4f}  {n}")