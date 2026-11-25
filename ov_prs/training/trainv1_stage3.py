#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train and evaluate the pairwise proposal ranker.

All 12 input features are available at inference. The loader also accepts
legacy 13-feature caches. Training saves each epoch and reports evaluation
metrics at the configured interval. Evaluation reads saved features.

    python -m ov_prs.training.trainv1_stage3 --stage collect
    python -m ov_prs.training.trainv1_stage3 --stage train
    python -m ov_prs.training.trainv1_stage3 --stage eval
    python -m ov_prs.training.trainv1_stage3 --stage collect --force_collect
"""
import argparse
import glob
import os
import random
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2

from ov_prs.config_panoworld import OUTPUT_DIR
from ov_prs.training.trainv1_stage1_2 import (
    build_models, CachedPanoWorldWrapper, APRSReferDataset,
    encode_proposals_with_clip, extract_topk_proposals,
    compute_iou, load_pw_feats, build_priors_for_batch, collate_fn,
    load_aligned_checkpoint, validate_aligned_checkpoint,
    SPLIT_BASE_TRAIN, SPLIT_BASE_TEST, SPLIT_NOVEL_TEST, SPLIT_OOD,
    CACHE_DIR, PW_FEAT_HW, DEVICE, IMAGE_SIZE, MIN_PRED_AREA,
)

# Include all nonempty ground-truth masks in the ranker evaluation.
MIN_MASK_AREA = 1

# ============ Settings ============
FEAT_DIM      = 12        # Ranker input size
LEGACY_FEAT_DIM = 13      # Legacy input size supported when loading cached features

TOPK_COLLECT = 10
CHUNK_SIZE   = 500

W_FG    = 0.3
W_AREA  = 0.2
W_POINT = 0.1

FEAT_DIR    = os.path.join(OUTPUT_DIR, "ranker_feats")
RANKER_PATH = os.path.join(OUTPUT_DIR, "ranker.pt")

COLLECT_SPLIT_ROOTS = {
    "base_train": SPLIT_BASE_TRAIN,
    "base_test":  SPLIT_BASE_TEST,
    "novel_test": SPLIT_NOVEL_TEST,
    "ood":        SPLIT_OOD,
}
EVAL_SPLITS = ["base_test", "novel_test", "ood"]
SPLIT_DIRS  = {s: os.path.join(FEAT_DIR, s) for s in COLLECT_SPLIT_ROOTS}


# ============ Feature compatibility ============
def _normalize_feat_dim(feats):
    """Convert cached features to FEAT_DIM by dropping index 10 from legacy inputs."""
    d = feats.shape[-1]
    if d == FEAT_DIM:
        return feats
    if d == LEGACY_FEAT_DIM:
        return torch.cat([feats[..., :10], feats[..., 11:]], dim=-1)
    raise RuntimeError(
        f"特征维度 {d} 不是 {FEAT_DIM} 或 {LEGACY_FEAT_DIM}，"
        f"检查 {FEAT_DIR} 是否被其他脚本覆盖")


# ============ Features available at inference ============
def _point_sim(mask, yaw, pitch, H, W, sigma=0.2):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    cy, cx = ys.mean(), xs.mean()
    prop_yaw = (cx / W) * 360.0 - 180.0
    prop_pitch = 90.0 - (cy / H) * 180.0
    dyaw = abs(prop_yaw - yaw)
    dyaw = min(dyaw, 360.0 - dyaw)
    dpitch = abs(prop_pitch - pitch)
    dist = np.sqrt(dyaw * dyaw + dpitch * dpitch) / 180.0
    return float(np.exp(-(dist * dist) / (2 * sigma * sigma)))


def _mask_point_overlap(mask, yaw, pitch, H, W, sigma=0.10):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    px_norm = ((yaw + 180.0) / 360.0) % 1.0
    py_norm = float(np.clip((90.0 - pitch) / 180.0, 0.0, 1.0))
    x_norm = xs.astype(np.float32) / W
    y_norm = ys.astype(np.float32) / H
    dx = np.abs(x_norm - px_norm)
    dx = np.minimum(dx, 1.0 - dx)
    dy = np.abs(y_norm - py_norm)
    dist2 = dx * dx + dy * dy
    weight = np.exp(-dist2 / (2 * sigma * sigma))
    return float(weight.mean())


def compute_proposal_features(
    proposal_masks_np, valid_props, topk_fg, cos_sim,
    yaw, pitch, H, W, min_area,
):
    """Compute the 12 inference-time features for each proposal.

      [0]  cos_sim       Cosine similarity to the image-conditioned text embedding
      [1]  topk_fg       Foreground confidence
      [2]  log_area      Log mask area fraction
      [3]  point_sim     Gaussian score for centroid-to-prior angular distance
      [4]  compactness   Mask compactness
      [5]  log1p(n_comp) Log of one plus the connected-component count
      [6]  rel_cos       Cosine score relative to the highest-scoring proposal
      [7]  rel_fg        Foreground score relative to the highest-scoring proposal
      [8]  log_aspect    Log bounding-box aspect ratio
      [9]  rel_area      Log area relative to the largest proposal
      [10] overlaps      Mean Gaussian point-prior weight across mask pixels
      [11] log_perims    Log of one plus the mask perimeter
    """
    K = len(proposal_masks_np)
    feats = np.zeros((K, FEAT_DIM), dtype=np.float32)
    areas = np.zeros(K, dtype=np.float32)
    point_sims = np.zeros(K, dtype=np.float32)
    log_perims = np.zeros(K, dtype=np.float32)
    overlaps = np.zeros(K, dtype=np.float32)

    for k in range(K):
        if not valid_props[k]:
            continue
        pm = proposal_masks_np[k]
        area = float(pm.sum())
        if area < min_area:
            continue
        areas[k] = area
        point_sims[k] = _point_sim(pm, yaw, pitch, H, W)
        overlaps[k] = _mask_point_overlap(pm, yaw, pitch, H, W)
        cnts, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        log_perims[k] = float(np.log(perim + 1.0))

    area_max = max(areas.max(), 1.0)
    cos_max = float(cos_sim.max()) if len(cos_sim) else 0.0
    fg_max = float(topk_fg.max()) if len(topk_fg) else 0.0

    for k in range(K):
        if not valid_props[k]:
            continue
        pm = proposal_masks_np[k]
        area = float(pm.sum())
        if area < min_area:
            continue

        area_ratio = area / (H * W)
        log_area = float(np.log(area_ratio + 1e-8))

        n_comp = cv2.connectedComponents(pm.astype(np.uint8))[0] - 1
        n_comp = max(n_comp, 1)

        cnts, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        compactness = (4 * np.pi * area) / (perim * perim + 1e-6)
        compactness = float(np.clip(compactness, 0.0, 1.0))

        ys, xs = np.nonzero(pm)
        bw = float(xs.max() - xs.min() + 1)
        bh = float(ys.max() - ys.min() + 1)
        log_aspect = float(np.log(bw / (bh + 1e-6) + 1e-6))

        rel_cos = float(cos_sim[k] - cos_max)
        rel_fg = float(topk_fg[k] - fg_max)
        rel_area = float(np.log(area / area_max + 1e-8))

        feats[k] = np.array([
            float(cos_sim[k]), float(topk_fg[k]), log_area,
            float(point_sims[k]), compactness,
            float(np.log(1.0 + n_comp)),
            rel_cos, rel_fg, log_aspect, rel_area,
            float(overlaps[k]), float(log_perims[k]),
        ], dtype=np.float32)
    return feats


def multi_signal_score_torch(feats, w_fg=W_FG, w_area=W_AREA, w_point=W_POINT):
    cos_norm = (feats[..., 0] + 1.0) / 2.0
    fg = feats[..., 1]
    area_p = torch.exp(-0.5 * ((feats[..., 2] - (-6.0)) / 1.5) ** 2)
    point = feats[..., 3]
    return cos_norm + w_fg * fg + w_area * area_p + w_point * point


# ============ Feature chunk storage ============
def _save_chunk(split_name, chunk_idx, feats_list, ious_list, valid_list):
    path = os.path.join(SPLIT_DIRS[split_name], f"part_{chunk_idx:04d}.pt")
    torch.save({
        "features": torch.stack(feats_list, 0),
        "ious":     torch.stack(ious_list, 0),
        "valid":    torch.stack(valid_list, 0),
    }, path)
    print(f"    [chunk {chunk_idx}] {len(feats_list)} 样本 -> {path}")
    return path


def _finalize_split(split_name, chunk_paths, n_total):
    manifest_path = os.path.join(SPLIT_DIRS[split_name], "manifest.pt")
    torch.save({
        "chunks": chunk_paths,
        "top_k": TOPK_COLLECT,
        "feat_dim": FEAT_DIM,
        "n_total": n_total,
        "split": split_name,
    }, manifest_path)
    print(f"    [manifest] {manifest_path}  chunks={len(chunk_paths)}  n={n_total}")


def split_has_features(split_name):
    return os.path.exists(os.path.join(SPLIT_DIRS[split_name], "manifest.pt"))


def load_split_features(split_name):
    """Load and normalize cached features, including legacy 13-feature inputs."""
    manifest_path = os.path.join(SPLIT_DIRS[split_name], "manifest.pt")
    if not os.path.exists(manifest_path):
        return None
    manifest = torch.load(manifest_path, map_location="cpu")
    all_f, all_i, all_v = [], [], []
    for c in manifest["chunks"]:
        d = torch.load(c, map_location="cpu")
        all_f.append(d["features"])
        all_i.append(d["ious"])
        all_v.append(d["valid"])
    feats = _normalize_feat_dim(torch.cat(all_f, 0))
    return feats, torch.cat(all_i, 0), torch.cat(all_v, 0), manifest


# ============ Stage: collect ============
@torch.no_grad()
def _collect_one_split(split_name, split_root, models, cache_wrapper, limit=None):
    (processor, model, prior_fusion, injector,
     clip_vision, clip_text, clip_tokenizer, mapper, visual_projection) = models

    model.eval(); prior_fusion.eval()
    mapper.eval(); clip_vision.eval()

    ds = APRSReferDataset(
        os.path.join(split_root, "annotations.csv"),
        split_root, processor, cache_wrapper, IMAGE_SIZE, False,
        min_mask_area=MIN_MASK_AREA)
    if limit:
        ds.df = ds.df.iloc[:limit].reset_index(drop=True)
    print(f"  {len(ds)} 样本，top-{TOPK_COLLECT}，FEAT_DIM={FEAT_DIM}，"
          f"MIN_MASK_AREA={MIN_MASK_AREA}")

    split_dir = SPLIT_DIRS[split_name]
    os.makedirs(split_dir, exist_ok=True)
    for old in glob.glob(os.path.join(split_dir, "part_*.pt")):
        os.remove(old)
    old_manifest = os.path.join(split_dir, "manifest.pt")
    if os.path.exists(old_manifest):
        os.remove(old_manifest)

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=False)

    chunk_feats, chunk_ious, chunk_valid = [], [], []
    chunk_paths = []
    chunk_idx = 0
    n_total = 0
    n_skipped_empty = 0

    for step, batch in enumerate(loader):
        if (step + 1) % 100 == 0:
            print(f"    {step+1}/{len(loader)} ({time.strftime('%H:%M:%S')})")

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

        topk_idx, topk_fg = extract_topk_proposals(outputs, TOPK_COLLECT)
        proposal_logits = F.interpolate(
            outputs.masks_queries_logits, size=IMAGE_SIZE,
            mode="bilinear", align_corners=False)
        proposal_masks = torch.sigmoid(proposal_logits) > 0.5
        proposal_masks_list = [proposal_masks[b] for b in range(B)]
        proposal_masks_np_batch = proposal_masks.cpu().numpy()

        v_props, valid_props = encode_proposals_with_clip(
            batch["erp_np"], proposal_masks_list, topk_idx,
            clip_vision, visual_projection, DEVICE)

        loaded = load_pw_feats(batch, cache_wrapper)
        if loaded is None:
            continue
        pw_feats, text_feat = loaded
        z = mapper(pw_feats, text_feat)
        z_n = F.normalize(z, dim=-1).unsqueeze(1)
        v_n = F.normalize(v_props, dim=-1)
        cos_sim = (z_n * v_n).sum(dim=-1)

        for b in range(B):
            gts = batch["mask_labels"][b].cpu().numpy().astype(bool)
            if len(gts) == 0 or gts[0].sum() == 0:
                n_skipped_empty += 1
                continue
            gt = gts[0]
            uname = batch["unique_name"][b]
            yaw, pitch, _, _ = cache_wrapper.load(uname)

            K = TOPK_COLLECT
            k_prop_masks = np.stack([
                proposal_masks_np_batch[b, int(topk_idx[b, k].item())]
                for k in range(K)
            ], axis=0)
            k_valid = valid_props[b].cpu().numpy().copy()
            k_fg = topk_fg[b].cpu().numpy().astype(np.float32)
            k_cos = cos_sim[b].cpu().numpy().astype(np.float32)

            feats = compute_proposal_features(
                k_prop_masks, k_valid, k_fg, k_cos,
                yaw, pitch, IMAGE_SIZE[0], IMAGE_SIZE[1], MIN_PRED_AREA)

            ious = np.full(K, -1.0, dtype=np.float32)
            for k in range(K):
                if not k_valid[k]:
                    continue
                pm = k_prop_masks[k]
                if pm.sum() < MIN_PRED_AREA:
                    k_valid[k] = False
                    continue
                ious[k] = compute_iou(pm, gt)

            chunk_feats.append(torch.from_numpy(feats))
            chunk_ious.append(torch.from_numpy(ious))
            chunk_valid.append(torch.from_numpy(k_valid))
            n_total += 1

        if len(chunk_feats) >= CHUNK_SIZE:
            p = _save_chunk(split_name, chunk_idx, chunk_feats, chunk_ious, chunk_valid)
            chunk_paths.append(p)
            chunk_feats, chunk_ious, chunk_valid = [], [], []
            chunk_idx += 1
            torch.cuda.empty_cache()

    if chunk_feats:
        p = _save_chunk(split_name, chunk_idx, chunk_feats, chunk_ious, chunk_valid)
        chunk_paths.append(p)
        chunk_idx += 1

    _finalize_split(split_name, chunk_paths, n_total)
    print(f"    n_total={n_total}  skipped_empty={n_skipped_empty}")


@torch.no_grad()
def stage_collect(models, cache_wrapper, limit=None, splits=None, force=False):
    if splits is None:
        splits = ["base_train", "base_test", "novel_test", "ood"]

    for split_name in splits:
        if split_name not in COLLECT_SPLIT_ROOTS:
            print(f"  [skip] 未知 split {split_name}")
            continue
        if not force and split_has_features(split_name):
            print(f"\n[collect] {split_name} 已有特征，跳过（--force_collect 强制重跑）")
            continue
        print(f"\n[collect] {split_name}")
        _collect_one_split(split_name, COLLECT_SPLIT_ROOTS[split_name],
                           models, cache_wrapper, limit)


# ============ Ranker ============
class RerankScorer(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pairwise_loss(scores, ious, valid, margin=0.05):
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        ious = ious.unsqueeze(0)
        valid = valid.unsqueeze(0)
    B, K = scores.shape
    total_loss = 0.0
    n_pairs = 0
    for b in range(B):
        for i in range(K):
            if not valid[b, i]:
                continue
            for j in range(i + 1, K):
                if not valid[b, j]:
                    continue
                di = float(ious[b, i] - ious[b, j])
                if abs(di) < margin:
                    continue
                if di > 0:
                    total_loss = total_loss + F.logsigmoid(scores[b, i] - scores[b, j])
                else:
                    total_loss = total_loss + F.logsigmoid(scores[b, j] - scores[b, i])
                n_pairs += 1
    if n_pairs == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return -total_loss / n_pairs


def topk_metrics_from_scores(scores, ious, valid, ks=(1, 3, 5, 10)):
    scores = scores.clone()
    scores[~valid] = -1e9
    order = scores.argsort(dim=-1, descending=True)
    sorted_ious = ious.gather(1, order)
    sorted_valid = valid.gather(1, order)

    oracle_ious = sorted_ious.clone()
    oracle_ious[~sorted_valid] = -1.0
    any_valid = sorted_valid.any(dim=-1)

    out = {}
    for k in ks:
        if k == 1:
            top_val = sorted_ious[:, 0]
            top_val = torch.where(sorted_valid[:, 0] & any_valid,
                                   top_val, torch.zeros_like(top_val))
        else:
            top_val = oracle_ious[:, :k].max(dim=-1).values
            top_val = torch.where(any_valid, top_val, torch.zeros_like(top_val))
        out[f"top{k}"] = float(top_val.mean())
    return out


# ============ Stage: train ============
def stage_train(epochs=30, lr=1e-3, seed=42, eval_every=5):
    tr = load_split_features("base_train")
    if tr is None:
        print("  ❌ 找不到 base_train 特征，先跑 --stage collect")
        return
    features, ious, valid, _ = tr
    N, K, D = features.shape
    print(f"  base_train: N={N}, K={K}, D={D}, "
          f"valid_ratio={valid.float().mean():.2%}")

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(features, ious, valid),
                              batch_size=16, shuffle=True, generator=g)

    monitors = {}
    for s in ["base_test", "novel_test", "ood"]:
        d = load_split_features(s)
        if d is None:
            print(f"  ⚠️ 找不到 {s} 特征，跳过该 split 的验证")
            continue
        f, i, v, _ = d
        monitors[s] = (f.to(DEVICE), i.to(DEVICE), v.to(DEVICE))
        print(f"  {s}: N={f.shape[0]}, D={f.shape[2]}")

    if "base_test" not in monitors:
        print("\n  ❌ 没有 base_test，无法选 best")
        print("     先跑：python -m ov_prs.training.trainv1_stage3 --stage collect "
              "--collect_splits base_test novel_test ood")
        return

    te_features, te_ious, te_valid = monitors["base_test"]

    ranker = RerankScorer(D).to(DEVICE)
    print(f"  Ranker 参数量: {sum(p.numel() for p in ranker.parameters()):,} "
          f"(in_dim={D})")

    opt = torch.optim.AdamW(ranker.parameters(), lr=lr, weight_decay=0.01)
    epoch_dir = os.path.join(OUTPUT_DIR, "stage3")
    os.makedirs(epoch_dir, exist_ok=True)
    print(f"\nTraining for {epochs} epochs (evaluation every {eval_every} epochs).")
    for ep in range(epochs):
        ranker.train()
        tr_loss = 0.0; tr_n = 0
        for fb, ib, vb in train_loader:
            fb = fb.to(DEVICE); ib = ib.to(DEVICE); vb = vb.to(DEVICE)
            scores = ranker(fb)
            loss = pairwise_loss(scores, ib, vb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item(); tr_n += 1
        tr_loss /= max(tr_n, 1)

        if (ep + 1) % eval_every == 0:
            ranker.eval()
            with torch.no_grad():
                te_scores = ranker(te_features)
                ranker_top1 = topk_metrics_from_scores(
                    te_scores, te_ious, te_valid, ks=(1,))["top1"]

                cos_top1 = topk_metrics_from_scores(
                    te_features[..., 0], te_ious, te_valid, ks=(1,))["top1"]
                multi_top1 = topk_metrics_from_scores(
                    multi_signal_score_torch(te_features), te_ious, te_valid,
                    ks=(1,))["top1"]

                monitor_parts = []
                for s in ["novel_test", "ood"]:
                    if s not in monitors:
                        continue
                    f, i, v = monitors[s]
                    ms = ranker(f)
                    s_top1 = topk_metrics_from_scores(ms, i, v, ks=(1,))["top1"]
                    tag = s.replace("_test", "")
                    monitor_parts.append(f"{tag}={s_top1:.4f}")
                monitor_str = "  " + "  ".join(monitor_parts) if monitor_parts else ""

            print(f"  Ep {ep+1:>3}  loss {tr_loss:.4f}  "
                  f"[base_test] cos={cos_top1:.4f}  multi={multi_top1:.4f}  "
                  f"ranker={ranker_top1:.4f}{monitor_str}")
        else:
            print(f"  Ep {ep+1:>3}  loss {tr_loss:.4f}")

        torch.save(ranker.state_dict(), os.path.join(epoch_dir, f"epoch{ep + 1}.pt"))

    torch.save(ranker.state_dict(), RANKER_PATH)
    print(f"Saved final ranker to {RANKER_PATH} after {epochs} epochs.")


# ============ Stage: eval ============
@torch.no_grad()
def stage_eval(limit=None):
    if not os.path.exists(RANKER_PATH):
        print(f"❌ 找不到 {RANKER_PATH}，先跑 --stage train")
        return

    ranker = RerankScorer(FEAT_DIM).to(DEVICE)
    state = torch.load(RANKER_PATH, map_location=DEVICE)
    try:
        ranker.load_state_dict(state)
    except RuntimeError as e:
        print(f"❌ ranker 维度不匹配：{e}")
        print(f"   当前脚本 ranker 输入维度 = {FEAT_DIM}，"
              f"但 {RANKER_PATH} 是旧版。请先 --stage train 重训。")
        return
    ranker.eval()
    print(f"  加载 ranker: {RANKER_PATH}  (in_dim={FEAT_DIM})")

    results = {}
    for split_name in EVAL_SPLITS:
        data = load_split_features(split_name)
        if data is None:
            print(f"  ⚠️ 找不到 {split_name} 特征，跳过")
            continue
        features, ious, valid, _ = data
        if limit:
            features = features[:limit]
            ious = ious[:limit]
            valid = valid[:limit]

        feats_dev = features.to(DEVICE)
        ious_dev = ious.to(DEVICE)
        valid_dev = valid.to(DEVICE)

        m_cos = topk_metrics_from_scores(feats_dev[..., 0], ious_dev, valid_dev,
                                          ks=(1, 3, 5, 10))
        m_multi = topk_metrics_from_scores(multi_signal_score_torch(feats_dev),
                                            ious_dev, valid_dev, ks=(1, 3, 5, 10))
        m_ranker = topk_metrics_from_scores(ranker(feats_dev), ious_dev, valid_dev,
                                             ks=(1, 3, 5, 10))

        results[split_name] = {
            "n": int(features.shape[0]),
            "cos": m_cos, "multi": m_multi, "ranker": m_ranker,
        }
        print(f"  {split_name}: n={features.shape[0]}  "
              f"cos={m_cos['top1']:.4f}  "
              f"multi={m_multi['top1']:.4f}  "
              f"ranker={m_ranker['top1']:.4f}")

    print("\n" + "=" * 94)
    print("明细：top1 / top3 oracle / top5 oracle / top10 oracle")
    print("=" * 94)
    print(f"{'split':<14}{'n':>6}{'method':<10}"
          f"{'top1':>10}{'top3':>10}{'top5':>10}{'top10':>10}")
    print("-" * 94)
    for split_name in results:
        r = results[split_name]
        for m in ("cos", "multi", "ranker"):
            d = r[m]
            print(f"{split_name:<14}{r['n']:>6}{m:<10}"
                  f"{d['top1']:>10.4f}{d['top3']:>10.4f}"
                  f"{d['top5']:>10.4f}{d['top10']:>10.4f}")
        print("-" * 94)

    if "base_test" in results and "novel_test" in results:
        print("\n" + "=" * 94)
        print("Base–Novel harmonic mean")
        print("=" * 94)
        b, n = results["base_test"], results["novel_test"]
        print(f"{'metric':<14}{'cos':>12}{'multi':>12}{'ranker':>14}{'Δ(r-c)':>12}")
        print("-" * 94)
        for k in ("top1", "top3", "top5", "top10"):
            H_cos = 2 * b["cos"][k] * n["cos"][k] / (b["cos"][k] + n["cos"][k] + 1e-8)
            H_mul = 2 * b["multi"][k] * n["multi"][k] / (b["multi"][k] + n["multi"][k] + 1e-8)
            H_rank = 2 * b["ranker"][k] * n["ranker"][k] / (b["ranker"][k] + n["ranker"][k] + 1e-8)
            print(f"H({k}){'':<8}{H_cos:>12.4f}{H_mul:>12.4f}"
                  f"{H_rank:>14.4f}{H_rank-H_cos:>+12.4f}")
        print("=" * 94)


# ============ Command-line entry point ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["collect", "train", "eval", "all"],
                        default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_collect", action="store_true")
    parser.add_argument("--collect_splits", nargs="+", choices=COLLECT_SPLIT_ROOTS)
    parser.add_argument("--checkpoint-dir", default=OUTPUT_DIR,
                        help="Root containing proposal and alignment checkpoints")
    args = parser.parse_args()
    if args.stage in ("train", "all") and (args.epochs < 1 or args.eval_every < 1):
        parser.error("--epochs and --eval_every must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("=" * 78)
    print("Learnable Pairwise Ranker Pipeline v1.2")
    print(f"  stage={args.stage}  limit={args.limit}  "
          f"epochs={args.epochs}  lr={args.lr}  "
          f"eval_every={args.eval_every}  seed={args.seed}")
    print(f"  MIN_MASK_AREA={MIN_MASK_AREA}（本脚本覆盖，主训练仍 30）")
    print(f"  FEAT_DIM={FEAT_DIM}")
    print(f"  FEAT_DIR={FEAT_DIR}")
    print(f"  RANKER_PATH={RANKER_PATH}")
    print("=" * 78)

    print("\n[特征状态]")
    for s in COLLECT_SPLIT_ROOTS:
        if split_has_features(s):
            d = load_split_features(s)
            n = d[0].shape[0] if d else 0
            d_dim = d[0].shape[2] if d else 0
            note = "" if d_dim == FEAT_DIM else f"（已适配为 {FEAT_DIM} 维）"
            print(f"  {s:<12} ✅ 已有 (n={n}, D={d_dim}){note}")
        else:
            print(f"  {s:<12} ❌ 缺")

    if args.stage in ("collect", "all"):
        print("\n[加载模型]")
        validate_aligned_checkpoint(args.checkpoint_dir)
        cache_wrapper = CachedPanoWorldWrapper(CACHE_DIR, PW_FEAT_HW)
        models = build_models(train_stage=2)
        load_aligned_checkpoint(models, args.checkpoint_dir)

        print("\n[Stage: collect]")
        stage_collect(models, cache_wrapper, limit=args.limit,
                      splits=args.collect_splits, force=args.force_collect)

    if args.stage in ("train", "all"):
        print("\n[Stage: train]")
        stage_train(epochs=args.epochs, lr=args.lr, seed=args.seed,
                    eval_every=args.eval_every)

    if args.stage in ("eval", "all"):
        print("\n[Stage: eval]")
        stage_eval(limit=args.limit)


if __name__ == "__main__":
    main()
