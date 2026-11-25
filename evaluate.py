#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate OV-PRS predictions on Base, Novel, and OOD with per-category reports."""

import argparse
from collections import defaultdict
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import cv2

from config_panoworld import OUTPUT_DIR
from trainv1_stage1_2 import (
    build_models, CachedPanoWorldWrapper, APRSReferDataset,
    encode_proposals_with_clip, extract_topk_proposals,
    compute_iou, load_pw_feats, build_priors_for_batch, collate_fn,
    load_aligned_checkpoint, validate_aligned_checkpoint,
    SPLIT_BASE_TEST, SPLIT_NOVEL_TEST, SPLIT_OOD,
    CACHE_DIR, PW_FEAT_HW, DEVICE, IMAGE_SIZE, MIN_PRED_AREA,
)
from trainv1_stage3 import FEAT_DIM, MIN_MASK_AREA, TOPK_COLLECT, RerankScorer, compute_proposal_features

SPLIT_MAP = {
    "base_test":  SPLIT_BASE_TEST,
    "novel_test": SPLIT_NOVEL_TEST,
    "ood":        SPLIT_OOD,
}

SAM3_BASELINE = {
    "75": {"base": 0.6219, "novel": 0.5014, "ood": 0.3967, "H": 0.5552},
    "90": {"base": 0.6133, "novel": 0.4976, "ood": 0.3954, "H": 0.5494},
}
# Paper SAM3 Macro IoU reference values for Novel and OOD.
SAM3_MACRO_IOU = {
    "novel": 0.4709,
    "ood":   0.3827,
}


# ============ Visualization helpers ============
def wrap_text(text, max_chars=60, max_lines=3):
    if not text:
        return [""]
    lines, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= max_chars:
            lines.append(cur)
            cur = ""
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines:
        last = lines[-1]
        if len(last) >= max_chars:
            lines[-1] = last[:max_chars - 3] + "..."
    return lines


def overlay_mask(erp_np, mask, color, alpha=0.55):
    out = erp_np.astype(np.float32).copy()
    if mask is not None and mask.sum() > 0:
        c = np.array(color, dtype=np.float32)
        out[mask] = out[mask] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_triptych(erp_np, pred_mask, gt_mask, info_lines, save_path):
    H, W = erp_np.shape[:2]
    img_orig = erp_np.copy()
    img_pred = overlay_mask(erp_np, pred_mask, (255, 60, 60))
    img_gt = overlay_mask(erp_np, gt_mask, (60, 220, 60))

    panels = [img_orig, img_pred, img_gt]
    labels = ["Input (ERP)", "Prediction", "GT"]

    header_h = 34 + 24 * len(info_lines)
    for p, lab in zip(panels, labels):
        cv2.rectangle(p, (0, 0), (W, header_h), (0, 0, 0), -1)
        cv2.putText(p, lab, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)
        for j, line in enumerate(info_lines):
            cv2.putText(p, line, (10, 50 + 22 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 220), 1, cv2.LINE_AA)

    sep = np.full((H, 4, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([panels[0], sep, panels[1], sep, panels[2]],
                             axis=1)
    Image.fromarray(canvas).save(save_path, quality=90)


# ============ Top-k statistics ============
def topk_stats(scores, ious, valid, ks=(1, 5, 10)):
    scores = scores.copy()
    scores[~valid] = -1e9
    order = scores.argsort()[::-1]

    top1 = 0.0
    for idx in order:
        if valid[idx]:
            top1 = float(ious[idx])
            break

    oracle = {}
    for k in ks:
        seen, best = 0, 0.0
        for idx in order:
            if not valid[idx]:
                continue
            if ious[idx] > best:
                best = float(ious[idx])
            seen += 1
            if seen >= k:
                break
        oracle[k] = best
    return top1, oracle


# ============ Evaluate one split ============
@torch.no_grad()
def evaluate_split(split_name, split_root, models, ranker, cache_wrapper,
                   results_dir, limit=None, save_vis=True, vis_max=100):
    (processor, model, prior_fusion, injector,
     clip_vision, clip_text, clip_tokenizer, mapper, visual_projection) = models

    model.eval(); prior_fusion.eval()
    mapper.eval(); clip_vision.eval()
    ranker.eval()

    # Read full referring expressions and categories from the CSV.
    csv_path = os.path.join(split_root, "annotations.csv")
    df_full = pd.read_csv(csv_path, encoding="utf-8-sig")
    uname_to_meta = {}
    for _, r in df_full.iterrows():
        u = str(r.get("unique_name", "")).strip()
        if not u:
            continue
        desc = str(r.get("description", "")).strip()
        if not desc or desc.lower() in ("nan", "none"):
            desc = str(r.get("target_word", r.get("core_word", ""))).strip()
        uname_to_meta[u] = {
            "category": str(r.get("category", "")).strip() or "unknown",
            "description": desc,
        }

    ds = APRSReferDataset(
        os.path.join(split_root, "annotations.csv"),
        split_root, processor, cache_wrapper, IMAGE_SIZE, False,
        min_mask_area=MIN_MASK_AREA)
    if limit:
        ds.df = ds.df.iloc[:limit].reset_index(drop=True)
    print(f"  {len(ds)} 样本")

    vis_dir = os.path.join(results_dir, split_name)
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=False)

    all_ranker_top1, all_cos_top1, all_fg_top1 = [], [], []
    all_ranker_o5, all_ranker_o10, all_cos_o5 = [], [], []
    all_hit = []
    n_skip_empty, vis_count = 0, 0

    per_cat_ranker     = defaultdict(list)
    per_cat_cos        = defaultdict(list)
    per_cat_fg         = defaultdict(list)
    per_cat_ranker_o5  = defaultdict(list)
    per_cat_ranker_o10 = defaultdict(list)
    per_cat_hit        = defaultdict(list)

    for step, batch in enumerate(loader):
        if (step + 1) % 50 == 0:
            cur = np.mean(all_ranker_top1) if all_ranker_top1 else 0.0
            print(f"    {step+1}/{len(loader)}  ranker_mIoU={cur:.4f}")

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

        # Use mapper-to-proposal cosine similarity.

        for b in range(B):
            gts = batch["mask_labels"][b].cpu().numpy().astype(bool)
            if len(gts) == 0 or gts[0].sum() == 0:
                n_skip_empty += 1
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

            # Build the 12-feature ranker input.
            feats = compute_proposal_features(
                k_prop_masks, k_valid, k_fg, k_cos,
                yaw, pitch, IMAGE_SIZE[0], IMAGE_SIZE[1], MIN_PRED_AREA)

            ious = np.full(K, -1.0, dtype=np.float32)
            for k in range(K):
                if not k_valid[k]:
                    continue
                if k_prop_masks[k].sum() < MIN_PRED_AREA:
                    k_valid[k] = False
                    continue
                ious[k] = compute_iou(k_prop_masks[k], gt)

            if not k_valid.any():
                continue

            ft = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)
            ranker_scores = ranker(ft).squeeze(0).cpu().numpy()

            ranker_top1, ranker_oracle = topk_stats(
                ranker_scores, ious, k_valid, ks=(5, 10))
            cos_top1, cos_oracle = topk_stats(
                k_cos, ious, k_valid, ks=(5,))
            fg_top1, _ = topk_stats(k_fg, ious, k_valid, ks=(1,))

            hit = 1.0 if ranker_top1 > 0.5 else 0.0

            all_ranker_top1.append(ranker_top1)
            all_cos_top1.append(cos_top1)
            all_fg_top1.append(fg_top1)
            all_ranker_o5.append(ranker_oracle[5])
            all_ranker_o10.append(ranker_oracle[10])
            all_cos_o5.append(cos_oracle[5])
            all_hit.append(hit)

            meta = uname_to_meta.get(uname, {})
            cat = meta.get("category", "") or "unknown"
            per_cat_ranker[cat].append(ranker_top1)
            per_cat_cos[cat].append(cos_top1)
            per_cat_fg[cat].append(fg_top1)
            per_cat_ranker_o5[cat].append(ranker_oracle[5])
            per_cat_ranker_o10[cat].append(ranker_oracle[10])
            per_cat_hit[cat].append(hit)

            if save_vis and (vis_max is None or vis_count < vis_max):
                valid_indices = np.where(k_valid)[0]
                best_local = ranker_scores[valid_indices].argmax()
                ranker_top1_idx = int(valid_indices[best_local])
                pred_mask = k_prop_masks[ranker_top1_idx]

                desc = meta.get("description", "")
                desc_lines = wrap_text(desc, max_chars=60, max_lines=2)
                info_lines = [f"cat: {cat}   |   IoU: {ranker_top1:.3f}"]
                for i, line in enumerate(desc_lines):
                    prefix = "q: " if i == 0 else "   "
                    info_lines.append(prefix + line)

                save_path = os.path.join(
                    vis_dir, f"{vis_count:04d}_{uname[:24]}.jpg")
                try:
                    make_triptych(batch["erp_np"][b], pred_mask, gt,
                                  info_lines, save_path)
                    vis_count += 1
                except Exception as e:
                    print(f"    [vis fail] {uname[:16]}: {e}")

    n = len(all_ranker_top1)
    if n == 0:
        return None

    per_category = {}
    for cat in per_cat_ranker:
        n_cat = len(per_cat_ranker[cat])
        per_category[cat] = {
            "n":           n_cat,
            "ranker_top1": float(np.mean(per_cat_ranker[cat])),   # = IoU_c
            "cos_top1":    float(np.mean(per_cat_cos[cat])),
            "fg_top1":     float(np.mean(per_cat_fg[cat])),
            "ranker_o5":   float(np.mean(per_cat_ranker_o5[cat])),
            "ranker_o10":  float(np.mean(per_cat_ranker_o10[cat])),
            "hit_rate":    float(np.mean(per_cat_hit[cat])),
        }

    # Instance-averaged IoU is the mean of top-1 IoUs.
    micro_iou = float(np.mean(all_ranker_top1))

    # Macro IoU is the mean of per-category IoUs.
    cat_ious_arr = np.array([m["ranker_top1"]
                              for m in per_category.values()])
    macro_iou = float(cat_ious_arr.mean()) if len(cat_ious_arr) > 0 else 0.0

    # Distribution of per-category IoUs.
    if len(cat_ious_arr) > 0:
        cat_iou_median = float(np.median(cat_ious_arr))
        cat_iou_min    = float(cat_ious_arr.min())
        cat_iou_max    = float(cat_ious_arr.max())
        cat_iou_std    = float(cat_ious_arr.std())
    else:
        cat_iou_median = cat_iou_min = cat_iou_max = cat_iou_std = 0.0

    results = {
        "split": split_name,
        "n": n,
        "n_skip_empty": n_skip_empty,
        "n_vis": vis_count,
        "num_categories": len(per_category),
        "micro_iou": micro_iou,     # = ranker_top1（sample-level avg）
        "macro_iou": macro_iou,     # category-level avg
        "ranker_top1": micro_iou,   # Retain the key used in older reports.
        "cos_top1": float(np.mean(all_cos_top1)),
        "fg_top1": float(np.mean(all_fg_top1)),
        "ranker_o5": float(np.mean(all_ranker_o5)),
        "ranker_o10": float(np.mean(all_ranker_o10)),
        "cos_o5": float(np.mean(all_cos_o5)),
        "hit_rate": float(np.mean(all_hit)),
        "cat_iou_median": cat_iou_median,
        "cat_iou_min": cat_iou_min,
        "cat_iou_max": cat_iou_max,
        "cat_iou_std": cat_iou_std,
        "per_category": per_category,
    }

    print(f"  [{split_name}] n={n} (skip_empty={n_skip_empty}, vis={vis_count})")
    print(f"    Micro IoU = {micro_iou:.4f}  "
          f"Macro IoU = {macro_iou:.4f}  "
          f"#categories = {len(per_category)}")
    print(f"    cos_top1={results['cos_top1']:.4f}  "
          f"fg_top1={results['fg_top1']:.4f}  "
          f"hit={results['hit_rate']*100:.2f}%")
    print(f"    category IoU  distribution:  "
          f"median={cat_iou_median:.4f}  "
          f"min={cat_iou_min:.4f}  max={cat_iou_max:.4f}  "
          f"std={cat_iou_std:.4f}")

    print(f"  [{split_name}] per-category (sorted by IoU_c desc):")
    for cat, m in sorted(per_category.items(),
                          key=lambda kv: -kv[1]["ranker_top1"]):
        print(f"      {cat:<24} n={m['n']:>4}  "
              f"IoU_c={m['ranker_top1']:.4f}  "
              f"cos={m['cos_top1']:.4f}  "
              f"fg={m['fg_top1']:.4f}  "
              f"hit={m['hit_rate']*100:5.1f}%  "
              f"o@5={m['ranker_o5']:.4f}  "
              f"o@10={m['ranker_o10']:.4f}")
    return results


# ============ Report ============
def write_report(all_results, report_path, ranker_path):
    L = []
    L.append("=" * 110)
    L.append("Ranker Final Test Report")
    L.append(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Ranker    : {ranker_path}  (in_dim={FEAT_DIM})")
    L.append("Model     : stage0/best proposals + stage2/best semantic alignment")
    L.append(f"Images    : {os.path.dirname(report_path)}/{{split}}/")
    L.append("=" * 110)
    L.append("")

    # ---------- Split summary ----------
    L.append("=" * 110)
    L.append("Table 1. Split-level summary")
    L.append("-" * 110)
    L.append(f"{'split':<14}{'n':>6}{'num_cat':>9}"
             f"{'micro_iou':>12}{'macro_iou':>12}"
             f"{'cos':>10}{'fg':>10}"
             f"{'r@5_or':>10}{'r@10_or':>10}{'hit':>9}")
    L.append("-" * 110)
    for name, r in all_results.items():
        if r is None:
            continue
        L.append(f"{name:<14}{r['n']:>6}{r['num_categories']:>9}"
                 f"{r['micro_iou']:>12.4f}{r['macro_iou']:>12.4f}"
                 f"{r['cos_top1']:>10.4f}{r['fg_top1']:>10.4f}"
                 f"{r['ranker_o5']:>10.4f}{r['ranker_o10']:>10.4f}"
                 f"{r['hit_rate']*100:>8.2f}%")
    L.append("=" * 110)
    L.append("")

    # ---------- Distribution of per-category IoUs ----------
    L.append("=" * 110)
    L.append("Table 2. Category-level IoU distribution")
    L.append("  micro_iou : sample-level mean of top-1 IoU")
    L.append("  macro_iou : mean of per-category IoU (Category-level Macro IoU)")
    L.append("-" * 110)
    L.append(f"{'split':<14}{'num_cat':>9}{'macro_iou':>12}"
             f"{'median':>10}{'min':>10}{'max':>10}{'std':>10}")
    L.append("-" * 110)
    for name, r in all_results.items():
        if r is None:
            continue
        L.append(f"{name:<14}{r['num_categories']:>9}"
                 f"{r['macro_iou']:>12.4f}"
                 f"{r['cat_iou_median']:>10.4f}"
                 f"{r['cat_iou_min']:>10.4f}"
                 f"{r['cat_iou_max']:>10.4f}"
                 f"{r['cat_iou_std']:>10.4f}")
    L.append("=" * 110)
    L.append("")

    # ---------- H_B,N ----------
    if "base_test" in all_results and "novel_test" in all_results:
        b = all_results["base_test"]; n = all_results["novel_test"]
        if b and n:
            H_ranker = 2 * b["micro_iou"] * n["micro_iou"] / (
                b["micro_iou"] + n["micro_iou"] + 1e-8)
            H_macro = 2 * b["macro_iou"] * n["macro_iou"] / (
                b["macro_iou"] + n["macro_iou"] + 1e-8)
            H_cos = 2 * b["cos_top1"] * n["cos_top1"] / (
                b["cos_top1"] + n["cos_top1"] + 1e-8)
            L.append("=" * 110)
            L.append("H_B,N")
            L.append("-" * 110)
            L.append(f"  H (ranker, micro) = {H_ranker:.4f}")
            L.append(f"  H (ranker, macro) = {H_macro:.4f}")
            L.append(f"  H (cos)           = {H_cos:.4f}")
            L.append("=" * 110)
            L.append("")

    # ---------- Full per-category breakdown ----------
    L.append("=" * 110)
    L.append("Table 3. Per-category breakdown")
    L.append("  IoU_c = mean top-1 IoU within this category")
    L.append("-" * 110)
    for split_name, r in all_results.items():
        if r is None or "per_category" not in r:
            continue
        pc = r["per_category"]
        if not pc:
            continue
        L.append("")
        L.append(f"[{split_name}]  total_n={r['n']}  "
                 f"num_categories={len(pc)}  "
                 f"Micro IoU={r['micro_iou']:.4f}  "
                 f"Macro IoU={r['macro_iou']:.4f}")
        L.append("-" * 110)
        L.append(f"  {'category':<24}{'n':>6}{'IoU_c':>10}"
                 f"{'cos':>10}{'fg':>10}"
                 f"{'hit':>10}{'o@5':>10}{'o@10':>10}")
        L.append("  " + "-" * 108)
        for cat, m in sorted(pc.items(),
                              key=lambda kv: -kv[1]["ranker_top1"]):
            L.append(f"  {cat:<24}{m['n']:>6}"
                     f"{m['ranker_top1']:>10.4f}"
                     f"{m['cos_top1']:>10.4f}"
                     f"{m['fg_top1']:>10.4f}"
                     f"{m['hit_rate']*100:>9.2f}%"
                     f"{m['ranker_o5']:>10.4f}"
                     f"{m['ranker_o10']:>10.4f}")
        L.append("  " + "-" * 108)
        # Summary rows.
        L.append(f"  {'MICRO (sample avg)':<24}{r['n']:>6}"
                 f"{r['micro_iou']:>10.4f}"
                 f"{r['cos_top1']:>10.4f}"
                 f"{r['fg_top1']:>10.4f}"
                 f"{r['hit_rate']*100:>9.2f}%"
                 f"{r['ranker_o5']:>10.4f}"
                 f"{r['ranker_o10']:>10.4f}")
        L.append(f"  {'MACRO (category avg)':<24}{'-':>6}"
                 f"{r['macro_iou']:>10.4f}"
                 f"{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>10}")
    L.append("=" * 110)
    L.append("")

    # ---------- Comparison with SAM3 ----------
    L.append("=" * 110)
    L.append("Compare to SAM3 baseline (main: best FOV 75°)")
    L.append(f"  SAM3 (75°): base=0.6219  novel=0.5014  ood=0.3967  H=0.5552")
    L.append(f"  SAM3 (90°): base=0.6133  novel=0.4976  ood=0.3954  H=0.5494")
    L.append(f"  SAM3 Macro IoU:  novel={SAM3_MACRO_IOU['novel']:.4f}  "
             f"ood={SAM3_MACRO_IOU['ood']:.4f}")
    L.append("")
    for split_key, base_key in [("base_test", "base"),
                                 ("novel_test", "novel"),
                                 ("ood", "ood")]:
        if split_key not in all_results or all_results[split_key] is None:
            continue
        v_micro = all_results[split_key]["micro_iou"]
        v_macro = all_results[split_key]["macro_iou"]
        for fov_tag in ["75", "90"]:
            b = SAM3_BASELINE[fov_tag][base_key]
            mark = "✓" if v_micro > b else "✗"
            L.append(f"  {base_key:<6} vs SAM3({fov_tag}°) [micro]: "
                     f"{v_micro:.4f} vs {b:.4f}  ({v_micro-b:+.4f})  {mark}")
        if base_key in SAM3_MACRO_IOU:
            bm = SAM3_MACRO_IOU[base_key]
            mark = "✓" if v_macro > bm else "✗"
            L.append(f"  {base_key:<6} vs SAM3 [macro]: "
                     f"{v_macro:.4f} vs {bm:.4f}  ({v_macro-bm:+.4f})  {mark}")
    if "base_test" in all_results and "novel_test" in all_results:
        b = all_results["base_test"]; n = all_results["novel_test"]
        if b and n:
            H = 2 * b["micro_iou"] * n["micro_iou"] / (
                b["micro_iou"] + n["micro_iou"] + 1e-8)
            for fov_tag in ["75", "90"]:
                bH = SAM3_BASELINE[fov_tag]["H"]
                mark = "✓" if H > bH else "✗"
                L.append(f"  H      vs SAM3({fov_tag}°): "
                         f"{H:.4f} vs {bH:.4f}  ({H-bH:+.4f})  {mark}")
    L.append("=" * 110)

    text = "\n".join(L)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n报告已保存: {report_path}")
    print("\n" + text)


# ============ Command-line entry point ============
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_MAP, default=list(SPLIT_MAP))
    parser.add_argument("--limit", type=int, help="Evaluate the first N examples in each split")
    parser.add_argument("--no_vis", action="store_true", help="Write the report without comparison images")
    parser.add_argument("--vis_max", type=int, help="Maximum comparison images per split")
    parser.add_argument("--checkpoint-dir", default=OUTPUT_DIR,
                        help="Root containing stage0/best, stage2/best and ranker.pt")
    parser.add_argument("--out-dir", default=os.path.join(OUTPUT_DIR, "results"))
    args = parser.parse_args()

    validate_aligned_checkpoint(args.checkpoint_dir)
    ranker_path = os.path.join(args.checkpoint_dir, "ranker.pt")
    ranker = RerankScorer(FEAT_DIM).to(DEVICE)
    ranker.load_state_dict(torch.load(ranker_path, map_location=DEVICE))
    ranker.eval()

    cache_wrapper = CachedPanoWorldWrapper(CACHE_DIR, PW_FEAT_HW)
    models = build_models(train_stage=2)
    load_aligned_checkpoint(models, args.checkpoint_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for split in args.splits:
        print(f"\n[Evaluate] {split}")
        results[split] = evaluate_split(
            split, SPLIT_MAP[split], models, ranker, cache_wrapper, args.out_dir,
            limit=args.limit, save_vis=not args.no_vis, vis_max=args.vis_max,
        )
    if not any(results.values()):
        raise RuntimeError("No examples could be scored. Check the benchmark files and cached features.")
    write_report(results, os.path.join(args.out_dir, "final_report.txt"), ranker_path)


if __name__ == "__main__":
    main()
