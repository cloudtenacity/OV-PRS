"""Precompute PanoWorld point priors and feature tensors for OV-PRS training."""

import argparse
import glob
import os
from pathlib import Path
import re
import tempfile

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ov_prs.config_panoworld import BENCHMARK_SPLITS, CACHE_DIR, DATA_ROOT, DO_SAMPLE, MAX_NEW_TOKENS, NUM_BEAMS
from ov_prs.load_panoworld import load_panoworld

PW_FEATURE_LAYERS = [2, 13, 24]
PW_FEAT_HW_CACHE = (15, 30)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_image_path(img_dir, filename, unique_name):
    p = os.path.join(img_dir, filename)
    if os.path.isfile(p): return p
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        p = os.path.join(img_dir, unique_name + ext)
        if os.path.isfile(p): return p
    stem = os.path.splitext(filename)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        p = os.path.join(img_dir, stem + ext)
        if os.path.isfile(p): return p
    for pat in [f"**/{filename}", f"**/{unique_name}.*", f"**/{stem}.*"]:
        ms = glob.glob(os.path.join(img_dir, pat), recursive=True)
        for match in ms:
            if os.path.isfile(match):
                return match
    return None


def image_for_row(split_dir, row, layout, image_indexes):
    unique_name = str(row["unique_name"]).strip()
    if layout == "legacy":
        return resolve_image_path(split_dir / "original", row["filename"], unique_name)

    folder = split_dir / "data" / str(row["category"]).strip() / "original"
    if folder not in image_indexes:
        image_indexes[folder] = {
            path.stem.lower(): path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        } if folder.is_dir() else {}
    image = image_indexes[folder].get(unique_name.lower())
    return str(image) if image is not None else None


def save_feature(feature, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
        temporary_path = Path(output.name)
        try:
            torch.save(feature, output)
            output.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def parse_point(resp):
    m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", resp)
    if m: return float(m.group(1)), float(m.group(2))
    ym = re.search(r"yaw[:\s]+(-?\d+\.?\d*)",   resp, re.I)
    pm = re.search(r"pitch[:\s]+(-?\d+\.?\d*)", resp, re.I)
    if ym and pm: return float(ym.group(1)), float(pm.group(1))
    m = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", resp)
    if m: return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


def feats_to_hw(f, target_hw):
    H_t, W_t = target_hw
    N_t = H_t * W_t
    if f.dim() == 2:
        f = f.unsqueeze(0).transpose(1, 2)
    elif f.dim() == 3:
        B, a, b = f.shape
        if a < b: f = f.transpose(1, 2)
    elif f.dim() != 4:
        raise ValueError(f"未知形状 {tuple(f.shape)}")
    if f.dim() == 3:
        N = f.shape[-1]
        if N == N_t:
            h, w = H_t, W_t
        else:
            h = int(round((N / 2) ** 0.5)); w = N // h
            if h * w != N:
                for h_ in range(int(N ** 0.5), 0, -1):
                    if N % h_ == 0: h, w = h_, N // h_; break
        f = f.reshape(f.shape[0], f.shape[1], h, w)
    if f.shape[-2:] != (H_t, W_t):
        f = F.interpolate(f.float(), size=(H_t, W_t),
                          mode='bilinear', align_corners=False)
    return f.half()


def _find_llm_last_layer(model):
    """Locate the last language-model layer across supported model layouts."""
    candidates = [
        ("model.language_model.model.layers",
         lambda m: m.model.language_model.model.layers),
        ("model.language_model.layers",
         lambda m: m.model.language_model.layers),
        ("language_model.model.layers",
         lambda m: m.language_model.model.layers),
        ("language_model.layers",
         lambda m: m.language_model.layers),
        ("model.model.layers",
         lambda m: m.model.model.layers),
        ("model.layers",
         lambda m: m.model.layers),
    ]
    for name, fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                print(f"  [LLM Hook] 定位到 {name}，共 {len(layers)} 层，取最后一层")
                return layers[-1]
        except AttributeError:
            continue
    raise RuntimeError("无法定位 LLM 层，请检查 model 结构")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--layout", choices=("benchmark", "legacy"), default="benchmark",
                        help="Use the four benchmark splits or the original train/test layout")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing features")
    args = parser.parse_args()

    splits = BENCHMARK_SPLITS if args.layout == "benchmark" else ("train", "test")
    required = {"unique_name", "description", "category" if args.layout == "benchmark" else "filename"}
    datasets = []
    for split in splits:
        split_dir = args.data_root / split
        csv_path = split_dir / "annotations.csv"
        if not csv_path.is_file():
            parser.error(f"missing annotations: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
        if not required.issubset(df.columns):
            parser.error(f"{csv_path} is missing columns: {sorted(required.difference(df.columns))}")
        datasets.append((split_dir, df))

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print("加载 PanoWorld...")
    cfg, proc, tok, model = load_panoworld()
    for p in model.parameters(): p.requires_grad = False
    model.eval()

    # ---------- Vision hooks ----------
    blocks = model.model.visual.blocks
    cache = {}
    hooks = []
    def make_hook(i):
        def fn(m, i_, o):
            if isinstance(o, tuple): o = o[0]
            cache[i] = o.detach()
        return fn
    for idx in PW_FEATURE_LAYERS:
        if idx < len(blocks):
            hooks.append(blocks[idx].register_forward_hook(make_hook(idx)))

    # ---------- Final language-model layer, prefill only ----------
    text_holder = {}
    last_layer = _find_llm_last_layer(model)

    def text_hook(m, inp, out):
        if "feat" in text_holder:
            return  # Capture only the first forward pass (prefill).
        hs = out[0] if isinstance(out, tuple) else out
        if not torch.is_tensor(hs) or hs.dim() != 3:
            return
        # Shape (B, L, D). Beam expansion can increase B. Use the first beam.
        text_holder["feat"] = hs[0, -1, :].detach().float().cpu()  # (D,)
        text_holder["L"] = int(hs.shape[1])

    hooks.append(last_layer.register_forward_hook(text_hook))

    text_dim = None

    failures = 0
    for split_dir, df in datasets:
        image_indexes = {}
        print(f"\n[{split_dir.name}] {len(df)} samples")

        for _, r in tqdm(df.iterrows(), total=len(df), desc=split_dir.name):
            uname = str(r["unique_name"]).strip()
            if not uname or Path(uname).name != uname or uname in {".", ".."}:
                print(f"Invalid unique_name in {split_dir.name}: {uname!r}")
                failures += 1
                continue
            out_path = args.cache_dir / f"{uname}.pt"
            if out_path.is_file() and not args.overwrite:
                continue
            img_path = image_for_row(split_dir, r, args.layout, image_indexes)
            if img_path is None:
                print(f"Missing image for {split_dir.name}/{uname}")
                failures += 1
                continue

            cache.clear()
            text_holder.clear()
            prompt = (
                f'Look at this 360-degree panorama. Locate the object described as: "{r["description"]}"\n'
                'Output ONLY the spherical coordinates as a JSON array: [yaw, pitch]\n'
                'where yaw is in degrees (-180 to 180), pitch is in degrees (-90 to 90).\n'
                'Output ONLY the array, nothing else.'
            )
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text",  "text": prompt},
            ]}]
            try:
                inputs = proc.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v)
                          for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=DO_SAMPLE, num_beams=NUM_BEAMS,
                    )
                trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], output_ids)]
                resp = proc.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
                yaw, pitch = parse_point(resp)

                feats_out = []
                for idx in PW_FEATURE_LAYERS:
                    f = cache.get(idx)
                    if f is None:
                        feats_out.append(None); continue
                    f = feats_to_hw(f, PW_FEAT_HW_CACHE).cpu().squeeze(0)
                    feats_out.append(f)

                text_feat = text_holder.get("feat")
                if text_feat is None:
                    raise RuntimeError("PanoWorld text features were not captured")
                text_feat = text_feat.squeeze(0).float()  # Save the contextual feature in float32.
                if text_dim is None:
                    text_dim = text_feat.shape[-1]
                    print(f"  [Text] dim = {text_dim}")
            except Exception as e:
                print(f"\n[{uname}] Failed: {e}")
                failures += 1
                continue

            if any(f is None for f in feats_out):
                print(f"\n[{uname}] Missing vision features")
                failures += 1
                continue

            save_feature({
                "yaw": float(yaw), "pitch": float(pitch),
                "feats": feats_out,                       # 3 × (1152, 15, 30) fp16
                "text_feat": text_feat,                   # (D_text,) fp32
                "meta": {"H": PW_FEAT_HW_CACHE[0], "W": PW_FEAT_HW_CACHE[1],
                         "layers": PW_FEATURE_LAYERS,
                         "text_dim": int(text_feat.shape[-1])},
            }, out_path)
            torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()
    files = list(args.cache_dir.glob("*.pt"))
    total_gb = sum(path.stat().st_size for path in files) / 1e9
    print(f"\n{len(files)} cached features ({total_gb:.2f} GB) in {args.cache_dir}")
    if failures:
        raise SystemExit(f"{failures} samples could not be cached. See the messages above.")

if __name__ == "__main__":
    main()
