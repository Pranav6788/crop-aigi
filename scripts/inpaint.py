import torch, json
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import time
import os
os.environ["HF_TOKEN"] = ""
os.environ["HUGGING_FACE_HUB_TOKEN"] = ""

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
).to("mps")
pipe.set_progress_bar_config(disable=True)

PROMPT = ("maize leaf with northern leaf blight, long narrow grey-green "
          "necrotic lesions running along the leaf blade, photorealistic, "
          "natural daylight, field conditions, sharp focus")
NEG = "cartoon, illustration, oversaturated, blurry, distorted, watermark, text"

out = Path("../data/P3"); out.mkdir(exist_ok=True)
log = []

for mp in sorted(Path("../data/masks").glob("*_mask.png")):
    src = Path("../data/P1") / f"{mp.stem.replace('_mask','')}.jpg"
    if not src.exists():
        print("missing source:", src.name); continue

    img  = Image.open(src).convert("RGB")
    mask = Image.open(mp).convert("L")
    seed = abs(hash(src.stem)) % (2**31)

    t = time.time()
    res = pipe(prompt=PROMPT, negative_prompt=NEG,
               image=img, mask_image=mask,
               guidance_scale=7.5, num_inference_steps=50,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    # SD works at 512 and returns 512. Without this resize, resolution
    # alone separates P3 from P1/P2 and the benchmark is worthless.
    res.resize(img.size, Image.LANCZOS).save(
        out / f"{src.stem}_P3.jpg", quality=95)

    log.append({"src": src.name, "seed": seed})
    print(f"{src.name}  {time.time()-t:.1f}s")

json.dump({"prompt": PROMPT, "negative": NEG, "guidance": 7.5,
           "steps": 50, "model": "stabilityai/stable-diffusion-2-inpainting",
           "items": log}, open(out / "generation_log.json", "w"), indent=2)