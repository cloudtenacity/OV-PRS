#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train OV-PRS proposal generation and semantic alignment.

The model takes an ERP panorama and a natural-language referring expression
and predicts a binary mask for the referred object or objects.

Architecture
    The mapper projects the final prefill token from PanoWorld to a
    768-dimensional CLIP embedding. The token represents the panorama and
    the referring expression. The vision branch encodes each proposal crop,
    projects its features, and pools within the mask. Both branches align to
    a frozen CLIP text encoder. Alignment validation scores proposals by the
    cosine similarity of their mapped language and region embeddings.

Training phases
    TRAIN_STAGE=0 trains Mask2Former adapters, decoder and head parameters
    and prior fusion. It selects checkpoints by oracle IoU on base_test.
    TRAIN_STAGE=1 freezes the proposal generator and visual projection.
    It trains the mapper with a positive-pair cosine loss and selects
    checkpoints by the geometric mean of Base and Novel cosine scores.
    TRAIN_STAGE=2 freezes Mask2Former and prior fusion. It trains the mapper,
    visual projection, and CLIP vision adapters with alignment, contrastive,
    and region losses. It selects checkpoints by IoU on base_test.

Alignment details
    Phase 1 normalizes the mapper output to unit length and aligns it to the
    CLIP embedding of the target word. PanoWorld produces a separate
    contextual feature for each referring expression.
    Phase 2 chooses the highest-IoU proposal from the top-k candidates. The
    positive embedding interpolates between the target region and the proposal
    from IoU 0.4 to 0.8. Negative sampling prioritizes low-IoU proposals
    and bounds their count with HARD_NEG_K and MAX_NEG_K.

Checkpoint initialization
    A fresh phase 0 uses the initial Mask2Former weights. Later phases use
    the proposal and fusion weights from stage0/best. A fresh phase 2
    initializes the mapper from stage1/best. A resumed phase loads its
    trained components from its own best checkpoint.

    RESUME_STAGE0=1 TRAIN_STAGE=0 python trainv1_stage1_2.py
    RESUME_STAGE1=1 TRAIN_STAGE=1 python trainv1_stage1_2.py
    RESUME_STAGE2=1 TRAIN_STAGE=2 python trainv1_stage1_2.py

    Set BEST_SCORE_OVERRIDE when resuming from a checkpoint whose previous
    best score is unavailable in best_meta.json.

Other options
    Set GC=0 with TRAIN_STAGE=0 to disable gradient checkpointing.
    Set SMOKE_TEST=1 for a short pipeline run.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import math
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig,
    CLIPVisionModel,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPModel,
)

from config_panoworld import CACHE_DIR, CLIP_PATH, DATA_ROOT, DEVICE, MASK2FORMER_PATH, OUTPUT_DIR

VERSION = "1.0"

# ============================================================
# Training settings
# ============================================================
PRETRAINED_PATH = MASK2FORMER_PATH
CLIP_VISION_PATH = CLIP_PATH

SPLIT_BASE_TRAIN = os.path.join(DATA_ROOT, "base_train")
SPLIT_BASE_TEST  = os.path.join(DATA_ROOT, "base_test")
SPLIT_NOVEL_TEST = os.path.join(DATA_ROOT, "novel_test")
SPLIT_OOD        = os.path.join(DATA_ROOT, "ood")

TRAIN_STAGE   = int(os.environ.get("TRAIN_STAGE", 2))
RESUME_STAGE0 = os.environ.get("RESUME_STAGE0", "0") == "1"
RESUME_STAGE1 = os.environ.get("RESUME_STAGE1", "0") == "1"
RESUME_STAGE2 = os.environ.get("RESUME_STAGE2", "0") == "1"
BEST_SCORE_OVERRIDE = os.environ.get("BEST_SCORE_OVERRIDE", "")

USE_GC = os.environ.get("GC", "1" if TRAIN_STAGE == 0 else "0") == "1"

assert TRAIN_STAGE in (0, 1, 2), f"TRAIN_STAGE must be 0/1/2, got {TRAIN_STAGE}"

STAGE_DIR = os.path.join(OUTPUT_DIR, f"stage{TRAIN_STAGE}")
VIS_DIR   = os.path.join(STAGE_DIR, "visualizations")
os.makedirs(STAGE_DIR, exist_ok=True)
os.makedirs(VIS_DIR,    exist_ok=True)

if TRAIN_STAGE == 0:
    BATCH_SIZE  = 4
    TOP_K_TRAIN = 10
elif TRAIN_STAGE == 1:
    BATCH_SIZE  = 16
    TOP_K_TRAIN = 10
elif TRAIN_STAGE == 2:
    BATCH_SIZE  = 2
    TOP_K_TRAIN = 5
TOP_K_EVAL = 10

NUM_EPOCHS          = 30
WARMUP_STEPS        = 100
WEIGHT_DECAY        = 0.05
IMAGE_SIZE          = (512, 1024)
NUM_CLASSES         = 1
LOG_EVERY           = 20

EVAL_EVERY_EPOCHS   = 2 if TRAIN_STAGE == 0 else 1
VIS_EVERY_EPOCHS    = 2 if TRAIN_STAGE == 0 else 1
SAVE_EVERY_EPOCHS   = 2 if TRAIN_STAGE == 0 else 1

EVAL_START_EPOCH = 1

EARLY_STOP_PATIENCE = 10
MIN_MASK_AREA       = 30
NUM_QUERIES         = 100

LR_MAPPER         = 1e-4
LR_CLIP_LORA      = 1e-5
LR_MASK2FORMER    = 5e-5
LR_PRIOR_FUSION   = 1e-4 if TRAIN_STAGE == 0 else 5e-5

LORA_R_CLIP       = 16
LORA_ALPHA_CLIP   = 32
LORA_TARGETS_CLIP = ["q_proj", "v_proj", "k_proj", "out_proj"]

LORA_R_M2F        = 16
LORA_ALPHA_M2F    = 32
LORA_DROPOUT_M2F  = 0.1
LORA_TARGETS_M2F  = ["q_proj", "k_proj", "v_proj", "o_proj"]

PW_FEATURE_LAYERS = [2, 13, 24]
PW_FEATURE_DIM    = 1152
PW_FEAT_HW        = (60, 120)
PW_TEXT_DIM       = 4096          # Hidden size of the last PanoWorld language-model layer

PRIOR_HIDDEN_CH = 256
POINT_SIGMA_X   = 0.15
POINT_SIGMA_Y   = 0.15
SCALE_INIT      = 1.386

LOSS_CLASS_WEIGHT      = 2.0
LOSS_MASK_WEIGHT       = 1.0
LOSS_DICE_WEIGHT       = 8.0
LOSS_NO_OBJECT_WEIGHT  = 0.3

ALIGN_WEIGHT      = 0.5
CLIP_ALIGN_WEIGHT = 1.0
REGION_WEIGHT     = 0.3
INFO_NCE_TEMP     = 0.07

CONTRAST_TEMP = 0.07

NEG_IOU_THRESH = 0.5
HARD_NEG_K     = 5
MAX_NEG_K      = 10

POS_IOU_LOW    = 0.4
POS_IOU_HIGH   = 0.8

PROPOSAL_PAD_RATIO = 0.15
CLIP_INPUT_SIZE    = 224

FG_THRESHOLD     = 0.2
MIN_PRED_AREA    = 10

VIS_NUM_SAMPLES_PER_SPLIT = 20
VIS_SAMPLE_SEEDS = {"base_test": 42, "novel_test": 43, "ood": 44}

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
if SMOKE_TEST:
    NUM_EPOCHS = 1
    BATCH_SIZE = 2
    MAX_TRAIN_SAMPLES = 8
    MAX_VAL_SAMPLES   = 4
    VIS_NUM_SAMPLES_PER_SPLIT = 4
    SAVE_EVERY_EPOCHS = 999
    EARLY_STOP_PATIENCE = 999
    EVAL_EVERY_EPOCHS = 1
    VIS_EVERY_EPOCHS = 1
    EVAL_START_EPOCH = 1
    print("⚠️ SMOKE_TEST 已启用")
else:
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES   = None


# ============================================================
# Utility functions
# ============================================================
def yaw_pitch_to_norm(yaw, pitch):
    px_norm = ((yaw + 180.0) / 360.0) % 1.0
    py_norm = float(np.clip((90.0 - pitch) / 180.0, 0.0, 1.0))
    return px_norm, py_norm


def build_point_attention(H, W, px_norm, py_norm, sigma_x, sigma_y, device):
    yy = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1) / H
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, -1) / W
    dx = (xx - px_norm).abs()
    dx = torch.min(dx, 1.0 - dx)
    dy = (yy - py_norm).abs()
    return torch.exp(-(dx ** 2) / (2 * sigma_x ** 2)
                     - (dy ** 2) / (2 * sigma_y ** 2))


def compute_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def crop_proposal_region(erp_np, mask_np, padding_ratio=0.15, out_size=224):
    ys, xs = np.nonzero(mask_np)
    if len(ys) == 0:
        return None, None

    H, W = mask_np.shape
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    ph = int((y2 - y1 + 1) * padding_ratio)
    pw = int((x2 - x1 + 1) * padding_ratio)

    y1 = max(0, y1 - ph)
    y2 = min(H - 1, y2 + ph)
    x1p = x1 - pw
    x2p = x2 + pw

    if x2p >= W:
        img_crop = np.concatenate(
            [erp_np[y1:y2 + 1, x1p % W:], erp_np[y1:y2 + 1, :x2p - W]], axis=1)
        mask_crop = np.concatenate(
            [mask_np[y1:y2 + 1, x1p % W:], mask_np[y1:y2 + 1, :x2p - W]], axis=1)
    elif x1p < 0:
        img_crop = np.concatenate(
            [erp_np[y1:y2 + 1, x1p:], erp_np[y1:y2 + 1, :x2p + 1]], axis=1)
        mask_crop = np.concatenate(
            [mask_np[y1:y2 + 1, x1p:], mask_np[y1:y2 + 1, :x2p + 1]], axis=1)
    else:
        img_crop = erp_np[y1:y2 + 1, x1p:x2p + 1]
        mask_crop = mask_np[y1:y2 + 1, x1p:x2p + 1]

    if img_crop.size == 0 or mask_crop.sum() == 0:
        return None, None

    h, w = img_crop.shape[:2]
    side = max(h, w)
    scale = out_size / side
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    img_crop = cv2.resize(img_crop, (new_w, new_h))
    mask_crop = cv2.resize(mask_crop.astype(np.uint8), (new_w, new_h),
                            interpolation=cv2.INTER_NEAREST)

    img_pad = np.zeros((out_size, out_size, 3), dtype=img_crop.dtype)
    mask_pad = np.zeros((out_size, out_size), dtype=np.uint8)
    y_off = (out_size - new_h) // 2
    x_off = (out_size - new_w) // 2
    img_pad[y_off:y_off + new_h, x_off:x_off + new_w] = img_crop
    mask_pad[y_off:y_off + new_h, x_off:x_off + new_w] = mask_crop

    return img_pad, mask_pad.astype(bool)


def mask_pooling(patch_feats, mask_small):
    mask_sum = mask_small.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return (patch_feats * mask_small.unsqueeze(-1)).sum(dim=1) / mask_sum


def preprocess_clip_crop(img_np, device, size=224):
    img = img_np.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
    img = (img - mean) / std
    return torch.from_numpy(img).to(device)


# ============================================================
# Save and restore checkpoints
# ============================================================
def _should_save_m2f_key(key):
    if "pixel_level_module.encoder.swin" in key:
        return False
    if "relative_position_index" in key:
        return False
    return True


def _load_peft_adapter_to_model(target_peft_model, adapter_dir, name="LoRA"):
    if hasattr(target_peft_model, "load_adapter"):
        try:
            target_peft_model.load_adapter(adapter_dir, adapter_name="default")
            print(f"  [Load] {name} from {adapter_dir} (via peft.load_adapter)")
            return True
        except Exception as e:
            print(f"  [Warn] load_adapter 失败: {e}, fallback 到 load_state_dict")

    import safetensors.torch as st
    sft = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")

    if os.path.exists(sft):
        state = st.load_file(sft)
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        print(f"  [Warn] No {name} adapter at {adapter_dir}")
        return False

    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    if not lora_state:
        print(f"  [Warn] No LoRA keys in {adapter_dir}")
        return False

    missing, _ = target_peft_model.load_state_dict(lora_state, strict=False)
    n_miss_lora = sum(1 for k in missing if "lora_" in k)
    print(f"  [Load] {name} from {adapter_dir} "
          f"({len(lora_state)} keys, {n_miss_lora} missing)")
    return True


def save_checkpoint(model, clip_vision, mapper, prior_fusion, visual_projection,
                    processor, clip_tokenizer, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)

    m2f_state = {}
    for k, v in model.state_dict().items():
        if _should_save_m2f_key(k):
            m2f_state[k] = v.detach().cpu()
    torch.save(m2f_state, os.path.join(ckpt_dir, "m2f_full_state.pt"))
    total_mb = sum(v.numel() * v.element_size() for v in m2f_state.values()) / 1e6
    print(f"    [Save] m2f_full_state.pt ({len(m2f_state)} keys, {total_mb:.1f} MB)")

    if clip_vision is not None and TRAIN_STAGE >= 2:
        clip_vision.save_pretrained(os.path.join(ckpt_dir, "clip_vision"))

    if clip_tokenizer is not None:
        clip_tokenizer.save_pretrained(os.path.join(ckpt_dir, "clip_tokenizer"))
    processor.save_pretrained(ckpt_dir)
    if mapper is not None:
        torch.save(mapper.state_dict(),             os.path.join(ckpt_dir, "mapper.pt"))
    if prior_fusion is not None:
        torch.save(prior_fusion.state_dict(),       os.path.join(ckpt_dir, "prior_fusion.pt"))
    if visual_projection is not None:
        torch.save(visual_projection.state_dict(),  os.path.join(ckpt_dir, "visual_projection.pt"))
    print(f"  [Save] {ckpt_dir}")


def _save_best_meta(ckpt_dir, best_score, score_name):
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {"best_score": float(best_score), "score_name": score_name}
    meta_path = os.path.join(ckpt_dir, "best_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"    [Save] best_meta.json (best_score={best_score:.4f}, name={score_name})")


def _load_best_score(ckpt_dir):
    meta_path = os.path.join(ckpt_dir, "best_meta.json")
    if not os.path.exists(meta_path):
        return None, None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("best_score", None), meta.get("score_name", None)
    except Exception:
        return None, None


def _has_any(ckpt_dir, filenames):
    if not os.path.isdir(ckpt_dir):
        return False
    return any(os.path.exists(os.path.join(ckpt_dir, f)) for f in filenames)


def load_checkpoint(ckpt_dir, clip_vision, mapper, prior_fusion,
                    visual_projection, model, train_stage,
                    modules=None):
    if modules is None:
        if train_stage == 0:
            modules = ["m2f", "prior_fusion"]
        elif train_stage == 1:
            modules = ["m2f", "prior_fusion", "mapper"]
        elif train_stage == 2:
            modules = ["m2f", "prior_fusion", "mapper",
                       "visual_projection", "clip_lora"]
        else:
            modules = ["m2f", "prior_fusion", "mapper",
                       "visual_projection", "clip_lora"]

    modules = set(modules)

    if "m2f" in modules:
        m2f_full_path = os.path.join(ckpt_dir, "m2f_full_state.pt")
        if os.path.exists(m2f_full_path):
            m2f_state = torch.load(m2f_full_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(m2f_state, strict=False)
            real_miss = [k for k in missing
                         if "pixel_level_module.encoder.swin" not in k
                         and "relative_position_index" not in k]
            print(f"  [Load] m2f_full_state from {m2f_full_path} "
                  f"({len(m2f_state)} keys, miss={len(real_miss)}, unexpected={len(unexpected)})")
            if real_miss:
                print(f"    ⚠️ Real missing keys (first 3): {real_miss[:3]}")
        else:
            m2f_adapter = os.path.join(ckpt_dir, "adapter_model.safetensors")
            if os.path.exists(m2f_adapter):
                print(f"  [Warn] m2f_full_state.pt 不存在，只加载 LoRA")
                _load_peft_adapter_to_model(model, ckpt_dir, name="M2F (LoRA only)")

    if "mapper" in modules:
        map_path = os.path.join(ckpt_dir, "mapper.pt")
        if os.path.exists(map_path) and mapper is not None:
            state = torch.load(map_path, map_location="cpu")
            try:
                mapper.load_state_dict(state)
                print(f"  [Load] mapper from {map_path}")
            except RuntimeError as e:
                print(f"  [Warn] mapper shape mismatch，跳过加载并重新初始化")
                print(f"         ({str(e).splitlines()[0] if str(e) else 'shape mismatch'})")

    if "prior_fusion" in modules:
        pf_path = os.path.join(ckpt_dir, "prior_fusion.pt")
        if os.path.exists(pf_path) and prior_fusion is not None:
            prior_fusion.load_state_dict(torch.load(pf_path, map_location="cpu"))
            print(f"  [Load] prior_fusion from {pf_path}")

    if "visual_projection" in modules:
        vp_path = os.path.join(ckpt_dir, "visual_projection.pt")
        if os.path.exists(vp_path) and visual_projection is not None:
            visual_projection.load_state_dict(torch.load(vp_path, map_location="cpu"))
            print(f"  [Load] visual_projection from {vp_path}")

    if "clip_lora" in modules and clip_vision is not None:
        clip_lora_dir = os.path.join(ckpt_dir, "clip_vision")
        if os.path.exists(os.path.join(clip_lora_dir, "adapter_config.json")):
            _load_peft_adapter_to_model(clip_vision, clip_lora_dir, name="CLIP LoRA")


def validate_aligned_checkpoint(checkpoint_dir):
    """Check the proposal and semantic checkpoints consumed by Stage 3."""
    proposal_dir = os.path.join(checkpoint_dir, "stage0", "best")
    alignment_dir = os.path.join(checkpoint_dir, "stage2", "best")
    required = [
        os.path.join(proposal_dir, "prior_fusion.pt"),
        os.path.join(alignment_dir, "mapper.pt"),
        os.path.join(alignment_dir, "visual_projection.pt"),
        os.path.join(alignment_dir, "clip_vision", "adapter_config.json"),
    ]
    proposals = ("m2f_full_state.pt", "adapter_model.safetensors")
    clip_adapter = ("adapter_model.safetensors", "adapter_model.bin")
    missing = [path for path in required if not os.path.isfile(path)]
    if not _has_any(proposal_dir, proposals):
        missing.append(f"{proposal_dir}/m2f_full_state.pt")
    if not _has_any(os.path.join(alignment_dir, "clip_vision"), clip_adapter):
        missing.append(f"{alignment_dir}/clip_vision/adapter_model.safetensors")
    if missing:
        raise FileNotFoundError("Missing OV-PRS checkpoint files: " + ", ".join(missing))
    return proposal_dir, alignment_dir


def load_aligned_checkpoint(models, checkpoint_dir):
    proposal_dir, alignment_dir = validate_aligned_checkpoint(checkpoint_dir)
    _, model, prior_fusion, _, clip_vision, _, _, mapper, visual_projection = models
    for folder, modules in (
        (proposal_dir, ["m2f", "prior_fusion"]),
        (alignment_dir, ["mapper", "visual_projection", "clip_lora"]),
    ):
        load_checkpoint(folder, clip_vision, mapper, prior_fusion,
                        visual_projection, model, train_stage=2, modules=modules)


def _is_resume_current_stage():
    flags = [RESUME_STAGE0, RESUME_STAGE1, RESUME_STAGE2]
    return flags[TRAIN_STAGE]


def _resolve_load_plan():
    plan = []
    resume = _is_resume_current_stage()

    if TRAIN_STAGE == 0:
        if resume:
            d = os.path.join(OUTPUT_DIR, "stage0", "best")
            if _has_any(d, ["m2f_full_state.pt"]):
                return [(d, ["m2f", "prior_fusion"])]
        return []

    d0 = os.path.join(OUTPUT_DIR, "stage0", "best")
    if _has_any(d0, ["m2f_full_state.pt"]):
        plan.append((d0, ["m2f", "prior_fusion"]))

    if TRAIN_STAGE == 1:
        if resume:
            d = os.path.join(OUTPUT_DIR, "stage1", "best")
            if os.path.exists(os.path.join(d, "mapper.pt")):
                plan.append((d, ["mapper"]))
    elif TRAIN_STAGE == 2:
        if resume:
            d = os.path.join(OUTPUT_DIR, "stage2", "best")
            if os.path.exists(os.path.join(d, "mapper.pt")):
                plan.append((d, ["mapper", "visual_projection", "clip_lora"]))
        else:
            d = os.path.join(OUTPUT_DIR, "stage1", "best")
            if os.path.exists(os.path.join(d, "mapper.pt")):
                plan.append((d, ["mapper"]))

    return plan


# ============================================================
# Cached PanoWorld features
# ============================================================
class CachedPanoWorldWrapper:
    def __init__(self, cache_dir, target_hw):
        self.cache_dir = cache_dir
        self.target_hw = target_hw
        files = glob.glob(os.path.join(cache_dir, "*.pt"))
        print(f"  [Cache] {len(files)} 文件 @ {cache_dir}")
        if len(files) == 0:
            raise RuntimeError(f"缓存为空: {cache_dir}")

    def has(self, name):
        return os.path.exists(os.path.join(self.cache_dir, f"{name}.pt"))

    def load(self, name):
        d = torch.load(os.path.join(self.cache_dir, f"{name}.pt"),
                       map_location="cpu")
        yaw, pitch = float(d["yaw"]), float(d["pitch"])
        feats = []
        for f in d["feats"]:
            f = f.float().unsqueeze(0)
            if f.shape[-2:] != self.target_hw:
                f = F.interpolate(f, size=self.target_hw,
                                  mode="bilinear", align_corners=False)
            feats.append(f)
        text_feat = d.get("text_feat", None)
        if text_feat is not None:
            text_feat = text_feat.float()   # (PW_TEXT_DIM,)
        return yaw, pitch, feats, text_feat


# ============================================================
# Mapper / PriorFusion / Injector
# ============================================================
class PanoWorldMapper(nn.Module):
    """Map the final PanoWorld prefill token into CLIP space.

    The token contains both image and referring-expression context. The
    projection retains that context without adding layer normalization.
    """
    def __init__(self, in_dim_per_layer, num_layers, hidden_dim, out_dim,
                 text_dim=PW_TEXT_DIM):
        super().__init__()
        self.text_dim = text_dim
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, feats_list=None, text_feat=None):
        if text_feat is None:
            raise RuntimeError(
                "PanoWorldMapper 需要 text_feat；请确认缓存包含 text_feat")
        return F.normalize(self.mlp(text_feat), dim=-1)


class PanoWorldPriorFusion(nn.Module):
    def __init__(self, in_dim_per_layer, num_layers, hidden_dim, out_channels):
        super().__init__()
        total_in = in_dim_per_layer * num_layers
        self.fuse = nn.Sequential(
            nn.Conv2d(total_in,    hidden_dim, 1),
            nn.GroupNorm(8,        hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim,  hidden_dim, 3, padding=1),
            nn.GroupNorm(8,        hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim,  out_channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(SCALE_INIT))

    def forward(self, feats_list, point_attn):
        x = torch.cat(feats_list, dim=1)
        out = self.fuse(x)
        if point_attn.dim() == 2:
            point_attn = point_attn.view(1, 1, *point_attn.shape)
        elif point_attn.dim() == 3:
            point_attn = point_attn.unsqueeze(1)
        out = out * (0.5 + 0.5 * point_attn)
        return out * torch.sigmoid(self.scale)


class PriorInjector:
    def __init__(self, model):
        self.model  = model
        self._prior = None
        decoder = model.base_model.model.model.pixel_level_module.decoder
        self._orig = decoder.forward

        def patched(features, *args, **kwargs):
            if self._prior is not None and len(features) > 0:
                target = features[0]
                H, W = target.shape[-2:]
                p = F.interpolate(self._prior, size=(H, W),
                                  mode="bilinear", align_corners=False)
                p = p.to(dtype=target.dtype, device=target.device)
                if p.shape[1] == target.shape[1]:
                    features = [target + p] + list(features[1:])
            return self._orig(features, *args, **kwargs)

        decoder.forward = patched

    def set_prior(self, prior):
        self._prior = prior


def get_backbone_first_stage_channels(config):
    bc = getattr(config, "backbone_config", None)
    if bc is None: return 128
    for attr in ["embed_dim", "hidden_size", "hidden_sizes"]:
        v = getattr(bc, attr, None)
        if v is not None:
            return v[0] if isinstance(v, (list, tuple)) else v
    return 128


# ============================================================
# Dataset
# ============================================================
class APRSReferDataset(Dataset):
    def __init__(self, csv_path, split_root, processor, cache_wrapper,
                 image_size=(512, 1024), augment=False, min_mask_area=MIN_MASK_AREA):
        self.processor  = processor
        self.image_size = image_size
        self.augment    = augment
        self.split_root = split_root
        self.min_mask_area = min_mask_area

        self._orig_index = {}
        self._mask_index = {}

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        rows = []
        for _, r in df.iterrows():
            cat   = str(r.get("category", "")).strip()
            uname = str(r.get("unique_name", "")).strip()
            if not cat or not uname:
                continue

            if cat not in self._orig_index:
                self._orig_index[cat] = self._scan(
                    os.path.join(split_root, "data", cat, "original"))
            if cat not in self._mask_index:
                self._mask_index[cat] = self._scan(
                    os.path.join(split_root, "data", cat, "mask"))

            img_p  = self._orig_index[cat].get(uname.lower())
            mask_p = self._mask_index[cat].get(uname.lower())
            if not img_p or not mask_p or not cache_wrapper.has(uname):
                continue

            core_word = ""
            for k in ["core_word", "target_word", "core", "keyword"]:
                if k in r and pd.notna(r[k]) and str(r[k]).strip():
                    core_word = str(r[k]).strip()
                    break
            if not core_word and "description" in r and pd.notna(r["description"]):
                core_word = str(r["description"]).strip().split(",")[0].split(" near ")[0].strip()

            rows.append({
                "unique_name": uname,
                "category":    cat,
                "core_word":   core_word,
                "_img_path":   img_p,
                "_mask_path":  mask_p,
            })

        self.df = pd.DataFrame(rows)
        print(f"  [Dataset] {os.path.basename(split_root):<14} 有效 {len(self.df)}/{len(df)}")

    @staticmethod
    def _scan(folder):
        idx = {}
        if not os.path.isdir(folder): return idx
        for name in os.listdir(folder):
            stem, ext = os.path.splitext(name)
            if ext.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                idx[stem.lower()] = os.path.join(folder, name)
        return idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        rgb = Image.open(r["_img_path"]).convert("RGB").resize(
            (self.image_size[1], self.image_size[0]), Image.BILINEAR)

        if self.augment:
            from torchvision import transforms
            rgb = transforms.ColorJitter(0.1, 0.1, 0.1)(rgb)

        erp_np = np.array(rgb)

        msk = np.array(Image.open(r["_mask_path"]).convert("L").resize(
            (self.image_size[1], self.image_size[0]), Image.NEAREST))
        msk = (msk > 127).astype(np.uint8)

        inputs = self.processor(images=rgb, return_tensors="pt", do_resize=True,
                                size={"height": self.image_size[0],
                                      "width": self.image_size[1]})
        pixel_values = inputs["pixel_values"].squeeze(0)

        H, W = msk.shape
        if msk.sum() >= self.min_mask_area:
            mask_labels  = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)
            class_labels = torch.zeros(1, dtype=torch.long)
        else:
            mask_labels  = torch.zeros((0, H, W), dtype=torch.float32)
            class_labels = torch.zeros((0,),     dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "mask_labels":  mask_labels,
            "class_labels": class_labels,
            "unique_name":  r["unique_name"],
            "core_word":    r["core_word"],
            "erp_np":       erp_np,
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], 0),
        "mask_labels":  [b["mask_labels"]  for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
        "unique_name":  [b["unique_name"]  for b in batch],
        "core_word":    [b["core_word"]    for b in batch],
        "erp_np":       [b["erp_np"]       for b in batch],
    }


# ============================================================
# Model initialization
# ============================================================
def build_models(train_stage=TRAIN_STAGE):
    print("\n加载 Mask2Former...")
    config = Mask2FormerConfig.from_pretrained(PRETRAINED_PATH)
    config.num_labels      = NUM_CLASSES
    config.num_queries     = NUM_QUERIES
    config.class_weight    = LOSS_CLASS_WEIGHT
    config.mask_weight     = LOSS_MASK_WEIGHT
    config.dice_weight     = LOSS_DICE_WEIGHT
    config.no_object_weight = LOSS_NO_OBJECT_WEIGHT

    processor = AutoImageProcessor.from_pretrained(PRETRAINED_PATH)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        PRETRAINED_PATH, config=config, ignore_mismatched_sizes=True,
    ).to(DEVICE)

    from peft import LoraConfig, get_peft_model
    lora_m2f = LoraConfig(
        r=LORA_R_M2F, lora_alpha=LORA_ALPHA_M2F,
        target_modules=LORA_TARGETS_M2F,
        lora_dropout=LORA_DROPOUT_M2F, bias="none",
    )
    model = get_peft_model(model, lora_m2f)

    if train_stage == 0:
        m2f_prefixes = [
            "lora_",
            "base_model.model.class_predictor",
            "base_model.model.model.pixel_level_module.decoder",
            "base_model.model.model.transformer_module.decoder",
            "base_model.model.model.transformer_module.level_embed",
        ]
    else:
        m2f_prefixes = []

    for name, p in model.named_parameters():
        p.requires_grad = any((name.startswith(pp) or pp in name)
                              for pp in m2f_prefixes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  M2F 可训练: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    if USE_GC and train_stage == 0:
        try:
            model.base_model.model.gradient_checkpointing_enable()
            print(f"  [GC] gradient checkpointing 已启用")
        except Exception as e:
            print(f"  [GC] 启用失败: {e}")

    injector = PriorInjector(model)

    stage1_ch = get_backbone_first_stage_channels(model.base_model.model.config)
    prior_fusion = PanoWorldPriorFusion(
        in_dim_per_layer=PW_FEATURE_DIM,
        num_layers=len(PW_FEATURE_LAYERS),
        hidden_dim=PRIOR_HIDDEN_CH,
        out_channels=stage1_ch,
    ).to(DEVICE)

    if train_stage in (1, 2):
        for p in prior_fusion.parameters():
            p.requires_grad = False
        print(f"  prior_fusion 冻结 (Stage {train_stage})")
    else:
        print(f"  prior_fusion 可训 (Stage {train_stage})")

    if train_stage >= 1:
        print(f"\n加载 CLIP vision: {CLIP_VISION_PATH}")
        clip_vision = CLIPVisionModel.from_pretrained(CLIP_VISION_PATH).to(DEVICE)

        if train_stage >= 2:
            lora_clip = LoraConfig(
                r=LORA_R_CLIP, lora_alpha=LORA_ALPHA_CLIP,
                target_modules=LORA_TARGETS_CLIP,
                lora_dropout=0.1, bias="none",
            )
            clip_vision = get_peft_model(clip_vision, lora_clip)
            for name, p in clip_vision.named_parameters():
                p.requires_grad = "lora_" in name
            trainable_clip = sum(p.numel() for p in clip_vision.parameters()
                                 if p.requires_grad)
            print(f"  CLIP LoRA 可训练: {trainable_clip:,} "
                  f"(r={LORA_R_CLIP}, targets={LORA_TARGETS_CLIP}, 0-init)")
        else:
            for p in clip_vision.parameters():
                p.requires_grad = False
            print(f"  CLIP 全部冻结 (Stage 1)")

        print(f"\n加载 CLIP text (教师)...")
        clip_text = CLIPTextModel.from_pretrained(CLIP_VISION_PATH).to(DEVICE)
        clip_text.eval()
        for p in clip_text.parameters():
            p.requires_grad = False
        clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_VISION_PATH)

        print(f"\n加载 visual_projection (从 CLIPModel)...")
        clip_full = CLIPModel.from_pretrained(CLIP_VISION_PATH)
        visual_projection = clip_full.visual_projection.to(DEVICE)
        clip_vision_dim = clip_vision.config.hidden_size
        clip_joint_dim  = visual_projection.out_features

        if train_stage == 1:
            for p in visual_projection.parameters():
                p.requires_grad = False
            print(f"  visual_projection 冻结 (Stage 1 只用 align loss)")
        else:
            for p in visual_projection.parameters():
                p.requires_grad = True
            print(f"  visual_projection: {clip_vision_dim} → {clip_joint_dim} "
                  f"(可训: {sum(p.numel() for p in visual_projection.parameters()):,})")
        del clip_full

        mapper = PanoWorldMapper(
            in_dim_per_layer=PW_FEATURE_DIM,
            num_layers=len(PW_FEATURE_LAYERS),
            hidden_dim=512,
            out_dim=clip_joint_dim,
            text_dim=PW_TEXT_DIM,
        ).to(DEVICE)
        n_mapper = sum(p.numel() for p in mapper.parameters())
        print(f"  Mapper 参数量: {n_mapper:,} (text_dim={PW_TEXT_DIM}，纯 text_proj)")
    else:
        clip_vision = None
        clip_text = None
        clip_tokenizer = None
        visual_projection = None
        mapper = None

    return (processor, model, prior_fusion, injector,
            clip_vision, clip_text, clip_tokenizer, mapper, visual_projection)


# ============================================================
# CLIP encoding
# ============================================================
@torch.no_grad()
def extract_topk_proposals(outputs, top_k):
    cp = torch.softmax(outputs.class_queries_logits, dim=-1)
    fg = cp[..., :-1].max(dim=-1).values
    topk = fg.topk(top_k, dim=-1)
    return topk.indices, topk.values


def encode_proposals_with_clip(erp_list, proposal_masks_list, topk_idx,
                                clip_vision, visual_projection, device=DEVICE):
    B, K = topk_idx.shape
    crop_imgs, crop_masks, valid = [], [], []

    for b in range(B):
        erp_np = erp_list[b]
        masks  = proposal_masks_list[b].cpu().numpy()
        for k in range(K):
            q_idx = int(topk_idx[b, k].item())
            m     = masks[q_idx]
            img_c, mask_c = crop_proposal_region(erp_np, m,
                                                  PROPOSAL_PAD_RATIO,
                                                  CLIP_INPUT_SIZE)
            if img_c is None:
                img_c   = np.zeros((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, 3), dtype=np.uint8)
                mask_c  = np.zeros((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE), dtype=bool)
                valid.append(False)
            else:
                valid.append(True)
            crop_imgs.append(img_c)
            crop_masks.append(mask_c)

    batch_imgs = torch.stack([preprocess_clip_crop(x, device) for x in crop_imgs], 0)

    feats = clip_vision(pixel_values=batch_imgs).last_hidden_state
    feats = feats[:, 1:]
    feats = visual_projection(feats)

    P = feats.shape[1]

    crop_masks_t = torch.from_numpy(np.stack(crop_masks, 0)).to(device).float()
    side = int(P ** 0.5)
    mask_small = F.adaptive_avg_pool2d(crop_masks_t.unsqueeze(1), (side, side)) \
                   .reshape(-1, P)

    v = mask_pooling(feats, mask_small)
    v = v.reshape(B, K, -1)

    valid_t = torch.tensor(valid, dtype=torch.bool, device=device).reshape(B, K)
    return v, valid_t


def encode_gt_with_clip(erp_list, gt_masks_list, clip_vision, visual_projection,
                         device=DEVICE):
    B = len(erp_list)
    crop_imgs, crop_masks, valid = [], [], []

    for b in range(B):
        erp_np = erp_list[b]
        gt_mask = gt_masks_list[b]
        img_c, mask_c = crop_proposal_region(
            erp_np, gt_mask, PROPOSAL_PAD_RATIO, CLIP_INPUT_SIZE)
        if img_c is None:
            img_c  = np.zeros((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, 3), dtype=np.uint8)
            mask_c = np.zeros((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE), dtype=bool)
            valid.append(False)
        else:
            valid.append(True)
        crop_imgs.append(img_c)
        crop_masks.append(mask_c)

    batch_imgs = torch.stack([preprocess_clip_crop(x, device) for x in crop_imgs], 0)
    feats = clip_vision(pixel_values=batch_imgs).last_hidden_state[:, 1:]
    feats = visual_projection(feats)

    P = feats.shape[1]
    masks_t = torch.from_numpy(np.stack(crop_masks, 0)).to(device).float()
    side = int(P ** 0.5)
    mask_small = F.adaptive_avg_pool2d(masks_t.unsqueeze(1), (side, side)).reshape(-1, P)

    v = mask_pooling(feats, mask_small)
    valid_t = torch.tensor(valid, dtype=torch.bool, device=device)
    return v, valid_t


def get_clip_text_embeddings(core_words, clip_text, clip_tokenizer,
                             device=DEVICE):
    prompts = [f"a photo of a {w}." if w else "a photo of something."
               for w in core_words]
    tok = clip_tokenizer(prompts, padding=True, truncation=True,
                          return_tensors="pt").to(device)
    with torch.no_grad():
        out = clip_text(**tok).pooler_output
    return out


# ============================================================
# Losses
# ============================================================
def compute_loss_align(z, text_emb):
    z_n = F.normalize(z, dim=-1)
    t_n = F.normalize(text_emb, dim=-1)
    return (1.0 - (z_n * t_n).sum(dim=-1)).mean()


def compute_loss_align_contrastive(z, text_emb_all, core_words,
                                    temp=CONTRAST_TEMP):
    B = z.shape[0]
    z_n = F.normalize(z, dim=-1)
    t_n = F.normalize(text_emb_all, dim=-1)
    sim = (z_n @ t_n.T) / temp
    labels = torch.arange(B, device=z.device)

    lower_words = [w.lower() for w in core_words]
    same_mask = torch.zeros((B, B), dtype=torch.bool, device=z.device)
    for i in range(B):
        for j in range(B):
            if i != j and lower_words[i] == lower_words[j]:
                same_mask[i, j] = True
    sim = sim.masked_fill(same_mask, -1e4)

    return F.cross_entropy(sim, labels)


def _select_negatives(gt_iou, valid, q_pos,
                       neg_thresh=NEG_IOU_THRESH,
                       min_k=HARD_NEG_K, max_k=MAX_NEG_K):
    K = gt_iou.shape[0]
    cand_mask = (gt_iou < neg_thresh) & valid
    if 0 <= q_pos < K:
        cand_mask[q_pos] = False
    cand_idx = torch.where(cand_mask)[0]

    if len(cand_idx) > max_k:
        ious_in_neg = gt_iou[cand_idx]
        keep = ious_in_neg.argsort()[:max_k]
        cand_idx = cand_idx[keep]

    elif len(cand_idx) < min_k:
        sorted_idx = gt_iou.argsort()
        existing = set(cand_idx.tolist())
        for idx in sorted_idx.tolist():
            if (idx == q_pos or idx in existing
                or not bool(valid[idx])):
                continue
            cand_idx = torch.cat(
                [cand_idx, torch.tensor([idx], device=gt_iou.device)])
            existing.add(idx)
            if len(cand_idx) >= min_k:
                break

    return cand_idx


def compute_clip_align_loss(v_gt, v_props, text_emb, valid_gt, valid_props,
                             gt_iou_per_prop,
                             neg_iou_thresh=NEG_IOU_THRESH,
                             hard_neg_k=HARD_NEG_K,
                             max_neg_k=MAX_NEG_K,
                             pos_iou_low=POS_IOU_LOW,
                             pos_iou_high=POS_IOU_HIGH):
    B, K, D = v_props.shape
    temp = INFO_NCE_TEMP

    v_gt_n    = F.normalize(v_gt, dim=-1)
    v_props_n = F.normalize(v_props, dim=-1)
    text_n    = F.normalize(text_emb, dim=-1)

    losses = []
    for b in range(B):
        if not bool(valid_gt[b]):
            continue

        q_pos  = int(gt_iou_per_prop[b].argmax().item())
        iou_pos = float(gt_iou_per_prop[b, q_pos].item())

        w_prop = max(0.0, min(1.0,
                              (iou_pos - pos_iou_low) / (pos_iou_high - pos_iou_low)))

        if bool(valid_props[b, q_pos]) and w_prop > 0:
            v_pos_n = (w_prop * v_props_n[b, q_pos]
                       + (1 - w_prop) * v_gt_n[b])
            v_pos_n = F.normalize(v_pos_n, dim=-1)
        else:
            v_pos_n = v_gt_n[b]

        neg_idx = _select_negatives(
            gt_iou_per_prop[b], valid_props[b], q_pos,
            neg_iou_thresh, hard_neg_k, max_neg_k)
        if len(neg_idx) == 0:
            continue

        sim_pos = (v_pos_n * text_n[b]).sum() / temp
        sim_neg = (v_props_n[b, neg_idx] * text_n[b]).sum(dim=-1) / temp

        logits = torch.cat([sim_pos.unsqueeze(0), sim_neg], dim=0)
        target = torch.zeros(1, dtype=torch.long, device=v_gt.device)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))

    if not losses:
        return torch.tensor(0.0, device=v_gt.device)
    return torch.stack(losses).mean()


def compute_region_loss_gt_based(z, v_props, v_gt, valid_props, valid_gt,
                                  gt_iou_per_prop,
                                  neg_iou_thresh=NEG_IOU_THRESH,
                                  hard_neg_k=HARD_NEG_K,
                                  max_neg_k=MAX_NEG_K):
    B, K, D = v_props.shape
    temp = INFO_NCE_TEMP

    z_n = F.normalize(z, dim=-1)
    v_gt_n = F.normalize(v_gt, dim=-1)
    v_props_n = F.normalize(v_props, dim=-1)

    losses = []
    for b in range(B):
        if not bool(valid_gt[b]):
            continue

        q_pos = int(gt_iou_per_prop[b].argmax().item())
        neg_idx = _select_negatives(
            gt_iou_per_prop[b], valid_props[b], q_pos,
            neg_iou_thresh, hard_neg_k, max_neg_k)
        if len(neg_idx) == 0:
            continue

        sim_pos = (z_n[b] * v_gt_n[b]).sum() / temp
        sim_neg = (z_n[b] * v_props_n[b, neg_idx]).sum(dim=-1) / temp

        logits = torch.cat([sim_pos.unsqueeze(0), sim_neg], dim=0)
        target = torch.zeros(1, dtype=torch.long, device=z.device)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))

    if not losses:
        return torch.tensor(0.0, device=z.device)
    return torch.stack(losses).mean()


# ============================================================
# Training helpers
# ============================================================
def build_priors_for_batch(batch, cache_wrapper, prior_fusion):
    priors = []
    for uname in batch["unique_name"]:
        if not cache_wrapper.has(uname):
            return None
        yaw, pitch, feats, _ = cache_wrapper.load(uname)
        feats = [f.to(DEVICE) for f in feats]
        px_n, py_n = yaw_pitch_to_norm(yaw, pitch)
        attn = build_point_attention(PW_FEAT_HW[0], PW_FEAT_HW[1], px_n, py_n,
                                     POINT_SIGMA_X, POINT_SIGMA_Y, DEVICE)
        priors.append(prior_fusion(feats, attn))
    return torch.cat(priors, dim=0)


def load_pw_feats(batch, cache_wrapper):
    # Return the visual layers and contextual language feature.
    feats_per_layer = [[], [], []]
    text_feats = []
    for uname in batch["unique_name"]:
        if not cache_wrapper.has(uname):
            return None
        _, _, feats, tf = cache_wrapper.load(uname)
        for i, f in enumerate(feats):
            feats_per_layer[i].append(f.to(DEVICE))
        if tf is None:
            raise RuntimeError(
                f"缓存 {uname} 缺少 text_feat，请用新版 cash_panoworld.py 重新生成")
        text_feats.append(tf.to(DEVICE))
    feats = [torch.cat(layer, dim=0) for layer in feats_per_layer]
    text_feat = torch.stack(text_feats, 0)   # (B, PW_TEXT_DIM)
    return feats, text_feat


# ============================================================
# Training step
# ============================================================
def train_step(batch, model, prior_fusion, injector, mapper, clip_vision,
               clip_text, clip_tokenizer, visual_projection,
               cache_wrapper, optimizer, scheduler):
    pix = batch["pixel_values"].to(DEVICE, non_blocking=True)
    mask_labels  = [m.to(DEVICE) for m in batch["mask_labels"]]
    class_labels = [c.to(DEVICE) for c in batch["class_labels"]]
    B = pix.shape[0]

    # Stage 1 aligns positive examples with the mapper.
    if TRAIN_STAGE == 1:
        loaded = load_pw_feats(batch, cache_wrapper)
        if loaded is None:
            return None
        pw_feats, text_feat = loaded
        z = mapper(pw_feats, text_feat)

        text_emb = get_clip_text_embeddings(batch["core_word"], clip_text,
                                             clip_tokenizer, DEVICE)
        loss_align = compute_loss_align(z, text_emb)

        total_loss = loss_align
        loss_dict = {"main": 0.0, "align": loss_align.item(),
                     "clip": 0.0, "region": 0.0}

        optimizer.zero_grad()
        total_loss.backward()

        params = []
        for g in optimizer.param_groups:
            params.extend(g["params"])
        bad = any(
            (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in params if p.grad is not None
        )
        if bad:
            optimizer.zero_grad()
            scheduler.step()
            return None

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_dict["total"] = total_loss.item()
        return loss_dict

    # Stages 0 and 2 run a Mask2Former forward pass.
    if all(m.shape[0] == 0 for m in mask_labels):
        return None

    prior = build_priors_for_batch(batch, cache_wrapper, prior_fusion)
    if prior is None:
        return None
    injector.set_prior(prior)

    try:
        if TRAIN_STAGE == 0:
            outputs = model(pixel_values=pix,
                            mask_labels=mask_labels,
                            class_labels=class_labels)
            main_loss = outputs.loss
        else:
            outputs = model(pixel_values=pix)
            main_loss = None
    finally:
        injector.set_prior(None)

    if TRAIN_STAGE == 0:
        total_loss = main_loss
        loss_dict = {"main": main_loss.item(),
                     "align": 0.0, "clip": 0.0, "region": 0.0}

    else:
        with torch.no_grad():
            topk_idx, _ = extract_topk_proposals(outputs, TOP_K_TRAIN)
            proposal_logits = F.interpolate(
                outputs.masks_queries_logits, size=IMAGE_SIZE,
                mode="bilinear", align_corners=False)
            proposal_masks = torch.sigmoid(proposal_logits) > 0.5
            proposal_masks_np = proposal_masks.cpu().numpy()

            gt_iou_per_topk = torch.zeros((B, TOP_K_TRAIN), device=DEVICE)
            for b in range(B):
                gt = mask_labels[b][0].cpu().numpy().astype(bool) if mask_labels[b].shape[0] > 0 else None
                if gt is None:
                    continue
                ious = np.zeros(TOP_K_TRAIN, dtype=np.float32)
                for k in range(TOP_K_TRAIN):
                    q = int(topk_idx[b, k].item())
                    ious[k] = compute_iou(proposal_masks_np[b, q], gt)
                gt_iou_per_topk[b] = torch.from_numpy(ious).to(DEVICE)

        erp_list = batch["erp_np"]
        v_props, valid_props = encode_proposals_with_clip(
            erp_list, [proposal_masks[b] for b in range(B)], topk_idx,
            clip_vision, visual_projection, DEVICE)

        gt_masks_np = []
        for b in range(B):
            if mask_labels[b].shape[0] > 0:
                gt_masks_np.append(mask_labels[b][0].cpu().numpy().astype(bool))
            else:
                gt_masks_np.append(np.zeros(IMAGE_SIZE, dtype=bool))
        v_gt, valid_gt = encode_gt_with_clip(
            erp_list, gt_masks_np, clip_vision, visual_projection, DEVICE)

        loaded = load_pw_feats(batch, cache_wrapper)
        if loaded is None:
            return None
        pw_feats, text_feat = loaded
        z = mapper(pw_feats, text_feat)

        text_emb = get_clip_text_embeddings(batch["core_word"], clip_text,
                                             clip_tokenizer, DEVICE)
        loss_align = compute_loss_align(z, text_emb)
        loss_clip = compute_clip_align_loss(
            v_gt, v_props, text_emb, valid_gt, valid_props, gt_iou_per_topk)
        loss_region = compute_region_loss_gt_based(
            z, v_props, v_gt, valid_props, valid_gt, gt_iou_per_topk)

        if TRAIN_STAGE == 2:
            total_loss = (ALIGN_WEIGHT * loss_align
                          + CLIP_ALIGN_WEIGHT * loss_clip
                          + REGION_WEIGHT * loss_region)
            main_val = 0.0
        else:
            total_loss = (main_loss
                          + ALIGN_WEIGHT * loss_align
                          + CLIP_ALIGN_WEIGHT * loss_clip
                          + REGION_WEIGHT * loss_region)
            main_val = main_loss.item() if main_loss is not None else 0.0

        loss_dict = {
            "main":   main_val,
            "align":  loss_align.item(),
            "clip":   loss_clip.item(),
            "region": loss_region.item(),
        }

    optimizer.zero_grad()
    total_loss.backward()

    params = []
    for g in optimizer.param_groups:
        params.extend(g["params"])
    bad = any(
        (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
        for p in params if p.grad is not None
    )
    if bad:
        optimizer.zero_grad()
        scheduler.step()
        return None

    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    scheduler.step()

    loss_dict["total"] = total_loss.item()
    return loss_dict


# ============================================================
# Evaluation
# ============================================================
def _save_train_state(model, prior_fusion, mapper, clip_vision):
    return {
        "model": model.training,
        "pf": prior_fusion.training,
        "mapper": mapper.training if mapper is not None else False,
        "clip": clip_vision.training if clip_vision is not None else False,
    }


def _restore_train_state(states, model, prior_fusion, mapper, clip_vision):
    if states["model"]: model.train()
    else: model.eval()
    if states["pf"]: prior_fusion.train()
    else: prior_fusion.eval()
    if mapper is not None:
        if states["mapper"]: mapper.train()
        else: mapper.eval()
    if clip_vision is not None:
        if states["clip"]: clip_vision.train()
        else: clip_vision.eval()


@torch.no_grad()
def evaluate_oracle(model, prior_fusion, injector, cache_wrapper, val_loader,
                    top_k=TOP_K_EVAL):
    train_states = _save_train_state(model, prior_fusion, None, None)

    model.eval(); prior_fusion.eval()
    all_best, all_hit, all_num = [], [], []

    try:
        for batch in val_loader:
            pix = batch["pixel_values"].to(DEVICE)
            B = pix.shape[0]

            prior = build_priors_for_batch(batch, cache_wrapper, prior_fusion)
            if prior is None:
                continue

            injector.set_prior(prior)
            try:
                outputs = model(pixel_values=pix)
            finally:
                injector.set_prior(None)

            topk_idx, _ = extract_topk_proposals(outputs, top_k)
            proposal_logits = F.interpolate(
                outputs.masks_queries_logits, size=IMAGE_SIZE,
                mode="bilinear", align_corners=False)
            proposal_masks = torch.sigmoid(proposal_logits) > 0.5
            proposal_masks_np = proposal_masks.cpu().numpy()

            for b in range(B):
                gts = batch["mask_labels"][b].cpu().numpy().astype(bool)
                if len(gts) == 0:
                    continue
                gt = gts[0]
                best_iou = 0.0
                for k in range(top_k):
                    q = int(topk_idx[b, k].item())
                    pm = proposal_masks_np[b, q]
                    if pm.sum() < MIN_PRED_AREA:
                        continue
                    iou = compute_iou(pm, gt)
                    if iou > best_iou:
                        best_iou = iou
                all_best.append(best_iou)
                all_hit.append(int(best_iou > 0.5))
                all_num.append(1)
    finally:
        _restore_train_state(train_states, model, prior_fusion, None, None)

    if not all_best:
        return 0.0, 0, 0, 0.0
    return (float(np.mean(all_best)), int(np.sum(all_hit)),
            int(np.sum(all_num)), 0.0)


@torch.no_grad()
def evaluate(model, prior_fusion, injector, mapper, clip_vision,
             clip_text, clip_tokenizer, visual_projection,
             cache_wrapper, val_loader, top_k=TOP_K_EVAL):
    train_states = _save_train_state(model, prior_fusion, mapper, clip_vision)

    model.eval(); prior_fusion.eval()
    if mapper is not None: mapper.eval()
    if clip_vision is not None: clip_vision.eval()

    all_iou, all_hit, all_num = [], [], []
    all_text_cos = []
    all_oracle = []

    try:
        for batch in val_loader:
            pix = batch["pixel_values"].to(DEVICE)
            B = pix.shape[0]

            prior = build_priors_for_batch(batch, cache_wrapper, prior_fusion)
            if prior is None:
                continue

            injector.set_prior(prior)
            try:
                outputs = model(pixel_values=pix)
            finally:
                injector.set_prior(None)

            topk_idx, _ = extract_topk_proposals(outputs, top_k)
            proposal_logits = F.interpolate(
                outputs.masks_queries_logits, size=IMAGE_SIZE,
                mode="bilinear", align_corners=False)
            proposal_masks = torch.sigmoid(proposal_logits) > 0.5
            proposal_masks_list = [proposal_masks[b] for b in range(B)]
            proposal_masks_np = proposal_masks.cpu().numpy()

            erp_list = batch["erp_np"]
            v_props, valid_props = encode_proposals_with_clip(
                erp_list, proposal_masks_list, topk_idx,
                clip_vision, visual_projection, DEVICE)

            loaded = load_pw_feats(batch, cache_wrapper)
            if loaded is None:
                continue
            pw_feats, text_feat = loaded
            z = mapper(pw_feats, text_feat)

            text_emb = get_clip_text_embeddings(batch["core_word"], clip_text,
                                                 clip_tokenizer, DEVICE)
            text_cos = F.cosine_similarity(z, text_emb, dim=-1).mean().item()
            all_text_cos.append(text_cos)

            z_n = F.normalize(z, dim=-1).unsqueeze(1)
            v_n = F.normalize(v_props, dim=-1)
            sim = (z_n * v_n).sum(dim=-1)

            for b in range(B):
                gts = batch["mask_labels"][b].cpu().numpy().astype(bool)
                if len(gts) == 0:
                    continue
                gt = gts[0]

                best_iou = 0.0
                for k in range(top_k):
                    q = int(topk_idx[b, k].item())
                    pm = proposal_masks_np[b, q]
                    if pm.sum() < MIN_PRED_AREA:
                        continue
                    iou = compute_iou(pm, gt)
                    if iou > best_iou:
                        best_iou = iou
                all_oracle.append(best_iou)

                sim_b = sim[b].clone()
                sim_b[~valid_props[b]] = -1e9
                k_best = int(sim_b.argmax().item())
                q_best = int(topk_idx[b, k_best].item())

                pred = proposal_masks[b, q_best].cpu().numpy()
                if pred.sum() < MIN_PRED_AREA:
                    mean_iou = 0.0; hit = 0
                else:
                    ious = [compute_iou(pred, g) for g in gts]
                    mean_iou = float(np.mean(ious))
                    hit = int((np.array(ious) > 0.5).sum())

                all_iou.append(mean_iou); all_hit.append(hit); all_num.append(len(gts))
    finally:
        _restore_train_state(train_states, model, prior_fusion, mapper, clip_vision)

    if not all_iou:
        return 0.0, 0, 0, 0.0, 0.0
    mean_text_cos = float(np.mean(all_text_cos)) if all_text_cos else 0.0
    mean_oracle = float(np.mean(all_oracle)) if all_oracle else 0.0
    return (float(np.mean(all_iou)), int(np.sum(all_hit)),
            int(np.sum(all_num)), mean_text_cos, mean_oracle)


def evaluate_all(model, prior_fusion, injector, mapper, clip_vision,
                 clip_text, clip_tokenizer, visual_projection,
                 cache_wrapper, val_loaders):
    results = {}
    for name, loader in val_loaders.items():
        if loader is None:
            results[name] = (0.0, 0, 0, 0.0, 0.0); continue
        if TRAIN_STAGE == 0:
            iou, hit, num, _ = evaluate_oracle(
                model, prior_fusion, injector, cache_wrapper, loader)
            results[name] = (iou, hit, num, 0.0, iou)
        else:
            results[name] = evaluate(
                model, prior_fusion, injector, mapper, clip_vision,
                clip_text, clip_tokenizer, visual_projection,
                cache_wrapper, loader)
    return results


# ============================================================
# Visualization
# ============================================================
def get_palette(n=256, seed=42):
    rng = np.random.RandomState(seed)
    c = rng.randint(60, 255, size=(n, 3)).astype(np.uint8)
    c[0] = 0
    return c


@torch.no_grad()
def visualize_sample(model, injector, prior_fusion, mapper, clip_vision,
                     visual_projection, cache_wrapper, processor,
                     img_path, mask_path, unique_name, save_path):
    rgb = Image.open(img_path).convert("RGB").resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    rgb_np = np.array(rgb)
    erp_np = rgb_np

    gt_sem = None
    if mask_path and os.path.exists(mask_path):
        gt_sem = np.array(Image.open(mask_path).convert("L").resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.NEAREST))
        gt_sem = (gt_sem > 127).astype(np.uint8)

    inputs = processor(images=rgb, return_tensors="pt", do_resize=True,
                       size={"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]})
    pix = inputs["pixel_values"].to(DEVICE)

    if not cache_wrapper.has(unique_name):
        print(f"    跳过: {unique_name}"); return

    yaw, pitch, feats, text_feat = cache_wrapper.load(unique_name)
    feats = [f.to(DEVICE) for f in feats]
    px_n, py_n = yaw_pitch_to_norm(yaw, pitch)
    attn = build_point_attention(PW_FEAT_HW[0], PW_FEAT_HW[1], px_n, py_n,
                                 POINT_SIGMA_X, POINT_SIGMA_Y, DEVICE)
    prior = prior_fusion(feats, attn)

    injector.set_prior(prior)
    try:
        outputs = model(pixel_values=pix)
    finally:
        injector.set_prior(None)

    topk_idx, _ = extract_topk_proposals(outputs, TOP_K_EVAL)
    proposal_logits = F.interpolate(
        outputs.masks_queries_logits, size=IMAGE_SIZE,
        mode="bilinear", align_corners=False)
    proposal_masks = torch.sigmoid(proposal_logits) > 0.5

    if TRAIN_STAGE == 0:
        gt = gt_sem.astype(bool) if gt_sem is not None else None
        best_k, best_iou = 0, 0.0
        if gt is not None:
            for k in range(TOP_K_EVAL):
                q = int(topk_idx[0, k].item())
                pm = proposal_masks[0, q].cpu().numpy()
                if pm.sum() < MIN_PRED_AREA:
                    continue
                iou = compute_iou(pm, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_k = k
        k_best = best_k
        score_str = f"iou={best_iou:.3f}"
    else:
        v_props, valid_props = encode_proposals_with_clip(
            [erp_np], [proposal_masks[0]], topk_idx,
            clip_vision, visual_projection, DEVICE)
        z = mapper(feats, text_feat.to(DEVICE).unsqueeze(0))
        z_n = F.normalize(z, dim=-1).unsqueeze(1)
        v_n = F.normalize(v_props, dim=-1)
        sim = (z_n * v_n).sum(dim=-1)[0]
        sim_b = sim.clone()
        sim_b[~valid_props[0]] = -1e9
        k_best = int(sim_b.argmax().item())
        score_str = f"sim={float(sim[k_best]):.3f}"

    q_best = int(topk_idx[0, k_best].item())
    pred_mask = proposal_masks[0, q_best].cpu().numpy()
    palette = get_palette(256)

    overlay = rgb_np.copy().astype(np.float32)
    if pred_mask.sum() >= MIN_PRED_AREA:
        col = palette[1].astype(np.float32)
        overlay[pred_mask] = overlay[pred_mask] * 0.45 + col * 0.55
    overlay = overlay.clip(0, 255).astype(np.uint8)

    panels = [rgb_np.copy(), overlay.copy()]
    labels = ["Image", f"Pred ({score_str}, q={q_best})"]

    if gt_sem is not None:
        gt_c = np.zeros((*gt_sem.shape, 3), dtype=np.uint8)
        gt_c[gt_sem > 0] = palette[2]
        gt_ov = (rgb_np.astype(np.float32) * 0.45
                 + gt_c.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)
        panels.append(gt_ov); labels.append("GT")

    for i, p in enumerate(panels):
        cv2.rectangle(p, (0, 0), (500, 34), (0, 0, 0), -1)
        cv2.putText(p, labels[i], (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    H, W = IMAGE_SIZE
    px = int(round(px_n * W)) % W
    py = int(round(py_n * H))
    py = max(0, min(H - 1, py))
    cv2.circle(panels[1], (px, py), 10, (255, 255, 0), 3)

    sep = np.full((H, 4, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([np.concatenate([p, sep], axis=1)
                             for p in panels[:-1]] + [panels[-1]], axis=1)
    Image.fromarray(canvas).save(save_path, quality=92)


def run_visualization(model, processor, injector, prior_fusion, mapper,
                      clip_vision, visual_projection, cache_wrapper,
                      dataset, indices, epoch, tag):
    if not indices: return
    train_states = _save_train_state(model, prior_fusion, mapper, clip_vision)

    model.eval(); prior_fusion.eval()
    if mapper is not None: mapper.eval()
    if clip_vision is not None: clip_vision.eval()

    d = os.path.join(VIS_DIR, f"epoch{epoch}_{tag}")
    os.makedirs(d, exist_ok=True)
    print(f"\n  ---- Epoch {epoch} Vis [{tag}] ({len(indices)}) ----")
    for i, idx in enumerate(indices):
        r = dataset.df.iloc[idx]
        p = os.path.join(d, f"img{i:02d}_{r['unique_name']}.jpg")
        try:
            visualize_sample(model, injector, prior_fusion, mapper, clip_vision,
                             visual_projection, cache_wrapper, processor,
                             r["_img_path"], r["_mask_path"],
                             r["unique_name"], p)
        except Exception as e:
            import traceback; print(f"    [{i}] fail: {e}"); traceback.print_exc()

    _restore_train_state(train_states, model, prior_fusion, mapper, clip_vision)
    print(f"  ---- Vis output: {d} ----\n")


# ============================================================
# Training loop
# ============================================================
def _set_train_modes(model, prior_fusion, mapper, clip_vision):
    if TRAIN_STAGE == 0:
        model.train()
        prior_fusion.train()
    else:
        model.eval()
        prior_fusion.eval()

    if mapper is not None:
        mapper.train()

    if clip_vision is not None:
        if TRAIN_STAGE >= 2:
            clip_vision.train()
        else:
            clip_vision.eval()


def train():
    print("=" * 60)
    print(f"PanoWorld + Mask2Former + CLIP Teacher  v{VERSION}")
    print(f"TRAIN_STAGE     = {TRAIN_STAGE}")
    print(f"USE_GC          = {USE_GC}")
    print(f"RESUME_STAGE0   = {RESUME_STAGE0}")
    print(f"RESUME_STAGE1   = {RESUME_STAGE1}")
    print(f"RESUME_STAGE2   = {RESUME_STAGE2}")
    print(f"BEST_OVERRIDE   = {BEST_SCORE_OVERRIDE if BEST_SCORE_OVERRIDE else '(none)'}")
    print(f"STAGE_DIR       = {STAGE_DIR}")
    print(f"BATCH_SIZE      = {BATCH_SIZE}")
    print(f"TOP_K_TRAIN     = {TOP_K_TRAIN}")
    print(f"EVAL_START_EPOCH= {EVAL_START_EPOCH}")
    print(f"POS_IOU_LOW/HIGH= {POS_IOU_LOW}/{POS_IOU_HIGH}")
    print(f"NEG_IOU/HARD/MAX= {NEG_IOU_THRESH}/{HARD_NEG_K}/{MAX_NEG_K}")
    print(f"PW_TEXT_DIM     = {PW_TEXT_DIM}")
    print(f"Loss weights: ALIGN={ALIGN_WEIGHT}, CLIP_ALIGN={CLIP_ALIGN_WEIGHT}, "
          f"REGION={REGION_WEIGHT}")
    print("=" * 60)

    cache_wrapper = CachedPanoWorldWrapper(CACHE_DIR, PW_FEAT_HW)

    (processor, model, prior_fusion, injector,
     clip_vision, clip_text, clip_tokenizer, mapper, visual_projection) = build_models()

    load_plan = _resolve_load_plan()
    if load_plan:
        for ckpt_dir, modules in load_plan:
            print(f"\n[Load Plan] 从 {ckpt_dir} 加载 {modules}")
            load_checkpoint(ckpt_dir, clip_vision, mapper, prior_fusion,
                            visual_projection, model,
                            train_stage=TRAIN_STAGE, modules=modules)
    else:
        print(f"\n[Start] Stage {TRAIN_STAGE} 从头开始训练")

    best_score = -1.0
    score_name_prev = None
    if _is_resume_current_stage():
        if BEST_SCORE_OVERRIDE:
            try:
                best_score = float(BEST_SCORE_OVERRIDE)
                print(f"  [Resume] best_score from BEST_SCORE_OVERRIDE = {best_score:.4f}")
            except ValueError:
                print(f"  [Resume] BEST_SCORE_OVERRIDE='{BEST_SCORE_OVERRIDE}' 无法解析为数字，忽略")
        else:
            meta_score, meta_name = _load_best_score(os.path.join(STAGE_DIR, "best"))
            if meta_score is not None:
                best_score = meta_score
                score_name_prev = meta_name
                print(f"  [Resume] best_score from best_meta.json = {best_score:.4f} "
                      f"(name={meta_name})")
            else:
                print(f"  [Resume] 无 best_meta.json 也无 BEST_SCORE_OVERRIDE，"
                      f"best_score 初始化为 -1.0（下一个 epoch 一定会覆盖 best）")
                print(f"  [Resume] 如需保护已有 best，请设置 "
                      f"BEST_SCORE_OVERRIDE=<当前 best 的 score>")

    print("\n构建数据集...")
    train_ds = APRSReferDataset(
        os.path.join(SPLIT_BASE_TRAIN, "annotations.csv"),
        SPLIT_BASE_TRAIN, processor, cache_wrapper, IMAGE_SIZE, augment=True)

    val_dss = {
        "base_test":  APRSReferDataset(
            os.path.join(SPLIT_BASE_TEST, "annotations.csv"),
            SPLIT_BASE_TEST, processor, cache_wrapper, IMAGE_SIZE, False),
        "novel_test": APRSReferDataset(
            os.path.join(SPLIT_NOVEL_TEST, "annotations.csv"),
            SPLIT_NOVEL_TEST, processor, cache_wrapper, IMAGE_SIZE, False),
        "ood":        APRSReferDataset(
            os.path.join(SPLIT_OOD, "annotations.csv"),
            SPLIT_OOD, processor, cache_wrapper, IMAGE_SIZE, False),
    }

    if MAX_TRAIN_SAMPLES:
        train_ds.df = train_ds.df.iloc[:MAX_TRAIN_SAMPLES].reset_index(drop=True)
    if MAX_VAL_SAMPLES:
        for k, ds in val_dss.items():
            if len(ds) > 0:
                ds.df = ds.df.iloc[:MAX_VAL_SAMPLES].reset_index(drop=True)

    if len(train_ds) == 0:
        print("❌ 训练集为空"); return

    nw = 0 if SMOKE_TEST else 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loaders = {k: (DataLoader(ds, batch_size=1, shuffle=False,
                                  num_workers=nw, collate_fn=collate_fn,
                                  pin_memory=True) if len(ds) > 0 else None)
                   for k, ds in val_dss.items()}

    param_groups = []

    if TRAIN_STAGE >= 1:
        param_groups.append(
            {"params": [p for p in mapper.parameters() if p.requires_grad],
             "lr": LR_MAPPER})

    if TRAIN_STAGE >= 2:
        vp_trainable = [p for p in visual_projection.parameters() if p.requires_grad]
        if vp_trainable:
            param_groups.append({"params": vp_trainable, "lr": LR_MAPPER})

    if TRAIN_STAGE == 0:
        m2f_params = [p for p in model.parameters() if p.requires_grad]
        if m2f_params:
            param_groups.append({"params": m2f_params, "lr": LR_MASK2FORMER})

    if TRAIN_STAGE == 0:
        pf_params = [p for p in prior_fusion.parameters() if p.requires_grad]
        if pf_params:
            param_groups.append({"params": pf_params, "lr": LR_PRIOR_FUSION})

    if TRAIN_STAGE >= 2:
        clip_lora_params = [p for p in clip_vision.parameters() if p.requires_grad]
        if clip_lora_params:
            param_groups.append({"params": clip_lora_params, "lr": LR_CLIP_LORA})

    all_params = []
    for g in param_groups:
        all_params.extend(g["params"])
    print(f"\n总可训练参数: {sum(p.numel() for p in all_params):,}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = NUM_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(max(1, WARMUP_STEPS))
        prog = float(step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    vis_by_split = {}
    print(f"\n[Vis] 每 split {VIS_NUM_SAMPLES_PER_SPLIT} 张")
    for name, ds in val_dss.items():
        if ds is None or len(ds) == 0: continue
        seed = VIS_SAMPLE_SEEDS.get(name, 42)
        rng = np.random.RandomState(seed)
        n = min(VIS_NUM_SAMPLES_PER_SPLIT, len(ds))
        vis_by_split[name] = sorted(rng.choice(len(ds), n, replace=False).tolist())
        print(f"      [{name}] seed={seed}, n={n}")

    _set_train_modes(model, prior_fusion, mapper, clip_vision)

    patience = 0

    for epoch in range(NUM_EPOCHS):
        total = {"main": 0.0, "align": 0.0, "clip": 0.0, "region": 0.0}
        valid_steps = 0

        for step, batch in enumerate(train_loader):
            res = train_step(batch, model, prior_fusion, injector, mapper,
                             clip_vision, clip_text, clip_tokenizer,
                             visual_projection, cache_wrapper,
                             optimizer, scheduler)
            if res is None: continue
            for k in total: total[k] += res.get(k, 0.0)
            valid_steps += 1

            if (step + 1) % LOG_EVERY == 0 or SMOKE_TEST:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
                      f"| Step {step+1}/{len(train_loader)} "
                      f"| main {res['main']:.3f} "
                      f"align {res['align']:.4f} "
                      f"clip {res['clip']:.4f} "
                      f"region {res['region']:.4f} "
                      f"| LR {scheduler.get_last_lr()[0]:.2e}")

        if valid_steps > 0:
            print(f"\nEpoch {epoch+1} | avg "
                  f"main {total['main']/valid_steps:.4f} "
                  f"align {total['align']/valid_steps:.4f} "
                  f"clip {total['clip']/valid_steps:.4f} "
                  f"region {total['region']/valid_steps:.4f} "
                  f"| steps {valid_steps}")

        if (epoch + 1) >= EVAL_START_EPOCH and (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
            r = evaluate_all(model, prior_fusion, injector, mapper,
                              clip_vision, clip_text, clip_tokenizer,
                              visual_projection, cache_wrapper, val_loaders)
            m_b, h_b, n_b, cos_b, o_b = r["base_test"]
            m_n, h_n, n_n, cos_n, o_n = r["novel_test"]
            m_o, h_o, n_o, cos_o, o_o = r["ood"]
            H = (2 * m_b * m_n / (m_b + m_n)) if (m_b + m_n) > 0 else 0.0

            if TRAIN_STAGE == 0:
                print(f"  [Val-Oracle] base={m_b:.4f} ({h_b}/{n_b}) "
                      f"| novel={m_n:.4f} ({h_n}/{n_n}) "
                      f"| ood={m_o:.4f} ({h_o}/{n_o}) | H={H:.4f}")
                score = m_b; score_name = "oracle_base"
            elif TRAIN_STAGE == 1:
                print(f"  [Val-CLIP] base={m_b:.4f} oracle={o_b:.4f} cos={cos_b:.3f} ({h_b}/{n_b}) "
                      f"| novel={m_n:.4f} oracle={o_n:.4f} cos={cos_n:.3f} ({h_n}/{n_n}) "
                      f"| ood={m_o:.4f} ({h_o}/{n_o}) | H={H:.4f}")
                score = math.sqrt(max(cos_b, 0) * max(cos_n, 0))
                score_name = "cos_geo"
            elif TRAIN_STAGE == 2:
                print(f"  [Val-CLIP] base={m_b:.4f} oracle={o_b:.4f} cos={cos_b:.3f} ({h_b}/{n_b}) "
                      f"| novel={m_n:.4f} oracle={o_n:.4f} cos={cos_n:.3f} ({h_n}/{n_n}) "
                      f"| ood={m_o:.4f} ({h_o}/{o_n}) | H={H:.4f}")
                score = m_b; score_name = "iou_base"

            if score > best_score:
                best_score = score; patience = 0
                if not SMOKE_TEST:
                    save_checkpoint(model, clip_vision, mapper, prior_fusion,
                                    visual_projection, processor, clip_tokenizer,
                                    os.path.join(STAGE_DIR, "best"))
                    _save_best_meta(os.path.join(STAGE_DIR, "best"),
                                    best_score, score_name)
                    print(f"  ✅ best {score_name}={score:.4f} -> {STAGE_DIR}/best")
            else:
                patience += 1
                print(f"  ⚠️ no improve ({patience}/{EARLY_STOP_PATIENCE})")
                if patience >= EARLY_STOP_PATIENCE:
                    print("⏹️ 早停"); break

        if (epoch + 1) >= EVAL_START_EPOCH and (epoch + 1) % VIS_EVERY_EPOCHS == 0:
            for name, ids in vis_by_split.items():
                run_visualization(model, processor, injector, prior_fusion,
                                  mapper, clip_vision, visual_projection,
                                  cache_wrapper, val_dss[name], ids,
                                  epoch+1, name)

        if not SMOKE_TEST and (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
            save_checkpoint(model, clip_vision, mapper, prior_fusion,
                            visual_projection, processor, clip_tokenizer,
                            os.path.join(STAGE_DIR, f"epoch{epoch+1}"))

    if not SMOKE_TEST:
        save_checkpoint(model, clip_vision, mapper, prior_fusion,
                        visual_projection, processor, clip_tokenizer,
                        os.path.join(STAGE_DIR, "final"))
        print(f"\n✅ 结束 | Best score={best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the proposal generator or semantic alignment.",
        epilog="Set TRAIN_STAGE=0 for proposals, 1 for mapper warm-up, or 2 for joint alignment.",
    )
    parser.parse_args()
    train()
