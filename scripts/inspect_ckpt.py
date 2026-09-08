import torch
ckpt = torch.load("weights/effort_clip_L14_trainOn_sdv14.pth", map_location="cpu")
sd = ckpt.get("state_dict", ckpt)

print("--- keys outside encoder.layers ---")
for k in sd:
    if "encoder.layers" not in k:
        print(f"{k:65s} {tuple(sd[k].shape)}")

print("\n--- top-level module names ---")
import collections
print(collections.Counter(k.split('.')[1] for k in sd))

# import sys, torch

# path = sys.argv[1]
# ckpt = torch.load(path, map_location="cpu")

# print("type:", type(ckpt))
# if isinstance(ckpt, dict):
#     print("top-level keys:", list(ckpt.keys())[:10])

# sd = ckpt.get("state_dict", ckpt)
# print(f"\n{len(sd)} tensors\n")

# for k, v in list(sd.items())[:25]:
#     print(f"{k:65s} {tuple(v.shape)}")

# print("\n--- classifier-like keys ---")
# for k in sd:
#     if any(t in k.lower() for t in ["head", "fc", "classifier", "logit", "proj"]):
#         print(f"{k:65s} {tuple(sd[k].shape)}")

# total = sum(v.numel() for v in sd.values())
# print(f"\ntotal parameters: {total/1e6:.1f}M")