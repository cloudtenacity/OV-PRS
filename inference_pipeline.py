#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run OV-PRS image or frame-by-frame video inference."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import tempfile
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from config_panoworld import MAX_NEW_TOKENS, DO_SAMPLE, NUM_BEAMS
from load_panoworld import load_panoworld

from trainv1_stage1_2 import (
    OUTPUT_DIR, build_models, load_aligned_checkpoint, validate_aligned_checkpoint,
    DEVICE, IMAGE_SIZE, MIN_PRED_AREA, PW_FEAT_HW, PW_FEATURE_LAYERS,
    yaw_pitch_to_norm, build_point_attention, extract_topk_proposals,
    encode_proposals_with_clip,
)
from trainv1_stage3 import FEAT_DIM, TOPK_COLLECT, RerankScorer, compute_proposal_features

# ================= Settings =================
PW_INPUT_W    = 2048   # PanoWorld input width
PW_INPUT_H    = 1024   # PanoWorld input height


def _safe(s):
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)[:30]


# ================= Pad / Resize =================
def pad_to_2to1(img_pil):
    """Pad the image to 2:1 and return it with a flag indicating padding."""
    W, H = img_pil.size
    if abs(W / H - 2.0) <= 0.01:
        return img_pil, False
    if W > H * 2.0:
        new_W, new_H = W, int(W / 2.0)
    else:
        new_W, new_H = int(H * 2.0), H
    padded = Image.new("RGB", (new_W, new_H), (0, 0, 0))
    padded.paste(img_pil, ((new_W - W) // 2, (new_H - H) // 2))
    return padded, True


def prepare_for_panoworld(img_pil):
    """Pad the image to 2:1, then resize it to 2048 × 1024."""
    img_pil, _ = pad_to_2to1(img_pil)
    if img_pil.size != (PW_INPUT_W, PW_INPUT_H):
        img_pil = img_pil.resize((PW_INPUT_W, PW_INPUT_H), Image.BILINEAR)
    return img_pil


# ================= PanoWorld helpers =================
def _find_llm_last_layer(model):
    candidates = [
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                return layers[-1]
        except AttributeError:
            pass
    raise RuntimeError("LLM layer not found")


def feats_to_hw(f, target_hw):
    H_t, W_t = target_hw
    N_t = H_t * W_t
    if f.dim() == 2:
        f = f.unsqueeze(0).transpose(1, 2)
    elif f.dim() == 3:
        B, a, b = f.shape
        if a < b:
            f = f.transpose(1, 2)
    if f.dim() == 3:
        N = f.shape[-1]
        if N == N_t:
            h, w = H_t, W_t
        else:
            h = int(round((N / 2) ** 0.5))
            w = N // h
            if h * w != N:
                for h_ in range(int(N ** 0.5), 0, -1):
                    if N % h_ == 0:
                        h, w = h_, N // h_
                        break
        f = f.reshape(f.shape[0], f.shape[1], h, w)
    if f.shape[-2:] != (H_t, W_t):
        f = F.interpolate(f.float(), size=(H_t, W_t), mode='bilinear',
                          align_corners=False)
    return f.half()


def parse_point(resp):
    m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", resp)
    if m:
        return float(m.group(1)), float(m.group(2))
    ym = re.search(r"yaw[:\s]+(-?\d+\.?\d*)", resp, re.I)
    pm = re.search(r"pitch[:\s]+(-?\d+\.?\d*)", resp, re.I)
    if ym and pm:
        return float(ym.group(1)), float(pm.group(1))
    m = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", resp)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


# ================= Inference pipeline =================
class Pipeline:
    def __init__(self, checkpoint_dir=OUTPUT_DIR):
        validate_aligned_checkpoint(checkpoint_dir)
        ranker_ckpt = os.path.join(checkpoint_dir, "ranker.pt")
        if not os.path.isfile(ranker_ckpt):
            raise FileNotFoundError(f"Missing OV-PRS ranker: {ranker_ckpt}")

        print("[Init 1/2] 加载 PanoWorld...")
        cfg, proc, tok, pw_model = load_panoworld()
        for p in pw_model.parameters():
            p.requires_grad = False
        pw_model.eval()
        self.pw_model = pw_model
        self.pw_proc = proc

        self.pw_blocks = pw_model.model.visual.blocks
        self.pw_feat_cache = {}
        self.pw_text_holder = {}
        self.pw_hooks = []

        def make_feat_hook(i):
            def fn(m, i_, o):
                if isinstance(o, tuple):
                    o = o[0]
                self.pw_feat_cache[i] = o.detach()
            return fn

        for idx in PW_FEATURE_LAYERS:
            self.pw_hooks.append(
                self.pw_blocks[idx].register_forward_hook(make_feat_hook(idx)))

        last_layer = _find_llm_last_layer(pw_model)

        def text_hook(m, inp, out):
            if "feat" in self.pw_text_holder:
                return
            hs = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(hs) or hs.dim() != 3:
                return
            self.pw_text_holder["feat"] = hs[0, -1, :].detach().float().cpu()

        self.pw_hooks.append(last_layer.register_forward_hook(text_hook))
        print("  ✓ PanoWorld ready")

        print("[Init 2/2] 加载 M2F + CLIP + Ranker...")
        models = build_models(train_stage=2)
        (self.processor, self.model, self.prior_fusion, self.injector,
         self.clip_vision, self.clip_text, self.clip_tokenizer,
         self.mapper, self.visual_projection) = models
        load_aligned_checkpoint(models, checkpoint_dir)
        for module in (self.model, self.prior_fusion, self.clip_vision,
                       self.clip_text, self.mapper, self.visual_projection):
            if module is not None:
                module.eval()

        self.ranker = RerankScorer(FEAT_DIM).to(DEVICE)
        state = torch.load(ranker_ckpt, map_location=DEVICE)
        try:
            self.ranker.load_state_dict(state)
        except RuntimeError as error:
            raise RuntimeError(
                f"{ranker_ckpt} is incompatible with the {FEAT_DIM}-feature ranker. "
                "Use the release checkpoint or train it with trainv1_stage3.py."
            ) from error
        self.ranker.eval()
        print("  ✓ M2F + Ranker ready\n")

    @torch.no_grad()
    def _run_panoworld(self, img_pil, query):
        img_pw = prepare_for_panoworld(img_pil)
        self.pw_feat_cache.clear()
        self.pw_text_holder.clear()

        prompt = (
            f'Look at this 360-degree panorama. Locate the object described as: "{query}"\n'
            'Output ONLY the spherical coordinates as a JSON array: [yaw, pitch]\n'
            'where yaw is in degrees (-180 to 180), pitch is in degrees (-90 to 90).\n'
            'Output ONLY the array, nothing else.'
        )
        with tempfile.TemporaryDirectory(prefix="ov-prs-") as temp_dir:
            tmp = os.path.join(temp_dir, "panorama.jpg")
            img_pw.save(tmp, quality=95)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": tmp},
                {"type": "text", "text": prompt},
            ]}]
            inputs = self.pw_proc.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: (v.to(self.pw_model.device) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}

            output_ids = self.pw_model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE, num_beams=NUM_BEAMS,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], output_ids)]
        resp = self.pw_proc.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        yaw, pitch = parse_point(resp)

        feats_out = []
        for idx in PW_FEATURE_LAYERS:
            f = self.pw_feat_cache.get(idx)
            if f is None:
                raise RuntimeError(f"Layer {idx} missing")
            feats_out.append(feats_to_hw(f, PW_FEAT_HW).cpu().squeeze(0))

        text_feat = self.pw_text_holder.get("feat")
        if text_feat is None:
            raise RuntimeError("text_feat missing")

        return yaw, pitch, feats_out, text_feat.float()

    @torch.no_grad()
    def _run_m2f(self, img_m2f, yaw, pitch, feats_list, text_feat):
        erp_np = np.array(img_m2f)
        inputs = self.processor(images=img_m2f, return_tensors="pt",
                                do_resize=True,
                                size={"height": IMAGE_SIZE[0],
                                      "width": IMAGE_SIZE[1]})
        pix = inputs["pixel_values"].to(DEVICE)

        px_n, py_n = yaw_pitch_to_norm(yaw, pitch)
        attn = build_point_attention(PW_FEAT_HW[0], PW_FEAT_HW[1],
                                     px_n, py_n, 0.15, 0.15, DEVICE)
        feats_dev = [f.float().unsqueeze(0).to(DEVICE) for f in feats_list]

        prior = self.prior_fusion(feats_dev, attn)
        self.injector.set_prior(prior)
        try:
            outputs = self.model(pixel_values=pix)
        finally:
            self.injector.set_prior(None)

        topk_idx, topk_fg = extract_topk_proposals(outputs, TOPK_COLLECT)
        proposal_logits = F.interpolate(
            outputs.masks_queries_logits, size=IMAGE_SIZE,
            mode="bilinear", align_corners=False)
        proposal_masks = torch.sigmoid(proposal_logits) > 0.5
        proposal_masks_np = proposal_masks.cpu().numpy()

        v_props, valid_props = encode_proposals_with_clip(
            [erp_np], [proposal_masks[0]], topk_idx,
            self.clip_vision, self.visual_projection, DEVICE)

        z = self.mapper(feats_dev, text_feat.to(DEVICE).unsqueeze(0))
        z_n = F.normalize(z, dim=-1).unsqueeze(1)
        v_n = F.normalize(v_props, dim=-1)
        cos_sim = (z_n * v_n).sum(dim=-1)

        K = TOPK_COLLECT
        k_prop_masks = np.stack([
            proposal_masks_np[0, int(topk_idx[0, k].item())]
            for k in range(K)
        ], axis=0)
        k_valid = valid_props[0].cpu().numpy().copy()
        k_fg = topk_fg[0].cpu().numpy().astype(np.float32)
        k_cos = cos_sim[0].cpu().numpy().astype(np.float32)

        # Build the 12-feature ranker input.
        feats_12d = compute_proposal_features(
            k_prop_masks, k_valid, k_fg, k_cos,
            yaw, pitch, IMAGE_SIZE[0], IMAGE_SIZE[1], MIN_PRED_AREA)

        ft = torch.from_numpy(feats_12d).float().unsqueeze(0).to(DEVICE)
        ranker_scores = self.ranker(ft).squeeze(0).cpu().numpy()

        ranker_scores_cp = ranker_scores.copy()
        ranker_scores_cp[~k_valid] = -1e9
        best_k = int(ranker_scores_cp.argmax())
        final_mask = k_prop_masks[best_k]

        return final_mask, {
            "best_k": best_k,
            "best_score": float(ranker_scores[best_k]),
        }

    @torch.no_grad()
    def run_frame(self, img_pil, query):
        """Return the ERP mask and ranking information for one frame."""
        yaw, pitch, feats, text_feat = self._run_panoworld(img_pil, query)
        img_m2f = img_pil.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        mask_small, info = self._run_m2f(img_m2f, yaw, pitch, feats, text_feat)
        info["yaw"] = yaw
        info["pitch"] = pitch
        return mask_small, info


# ================= Visualization =================
def render_overlay(frame_rgb, mask_small, out_w, out_h, query, score=None):
    """Resize the mask to the output dimensions and overlay it on the RGB frame."""
    mask = cv2.resize(
        mask_small.astype(np.uint8), (out_w, out_h),
        interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = frame_rgb.copy().astype(np.float32)
    col = np.array([255, 60, 60], dtype=np.float32)
    if mask.sum() > 0:
        overlay[mask] = overlay[mask] * 0.55 + col * 0.45
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    cv2.rectangle(overlay, (0, 0), (out_w, 40), (0, 0, 0), -1)
    tag = f"Q: {query}"
    if score is not None:
        tag += f"   (score={score:.2f})"
    cv2.putText(overlay, tag, (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


# ================= Image mode =================
def process_image(pipeline, img_path, query, out_img):
    print(f"[Image] {img_path}")
    img_pil = Image.open(img_path).convert("RGB")

    t0 = time.time()
    mask_small, info = pipeline.run_frame(img_pil, query)
    print(f"  yaw={info['yaw']:.2f} pitch={info['pitch']:.2f}  "
          f"best_k={info['best_k']}  score={info['best_score']:.4f}  "
          f"({time.time()-t0:.2f}s)")

    img_m2f = img_pil.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    erp_np = np.array(img_m2f)
    H, W = IMAGE_SIZE
    overlay_bgr = render_overlay(erp_np, mask_small, W, H, query,
                                 info["best_score"])
    canvas = np.concatenate(
        [cv2.cvtColor(erp_np, cv2.COLOR_RGB2BGR),
         np.ones((H, 10, 3), dtype=np.uint8) * 255,
         overlay_bgr], axis=1)
    cv2.imwrite(out_img, canvas)
    print(f"  ✅ saved: {out_img}")


# ================= Video mode =================
def process_video(pipeline, video_path, query, out_video, out_frames_dir,
                  vis_every, max_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_frames:
        n_total = min(n_total, max_frames)

    # Pad the output frames to a 2:1 aspect ratio.
    if abs(w_in / h_in - 2.0) > 0.01:
        if w_in / h_in > 2.0:
            out_w, out_h = w_in, int(w_in / 2.0)
        else:
            out_w, out_h = int(h_in * 2.0), h_in
        need_pad = True
    else:
        out_w, out_h = w_in, h_in
        need_pad = False

    print(f"[Video] {video_path}")
    print(f"  frames={n_total}  fps={fps:.1f}  "
          f"in={w_in}x{h_in}  out={out_w}x{out_h}  "
          f"pad={'yes' if need_pad else 'no'}")

    os.makedirs(out_frames_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"Could not write video: {out_video}")

    y_off = (out_h - h_in) // 2 if need_pad else 0
    x_off = (out_w - w_in) // 2 if need_pad else 0

    t0 = time.time()
    vis_count = 0
    frame_id = 0

    try:
        while frame_id < n_total:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if need_pad:
                padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                padded[y_off:y_off + h_in, x_off:x_off + w_in] = frame_bgr
                frame_bgr = padded

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            try:
                mask_small, info = pipeline.run_frame(img_pil, query)
            except Exception as e:
                print(f"    [Warn] frame {frame_id} failed: {e}")
                writer.write(frame_bgr)
                frame_id += 1
                continue

            out_bgr = render_overlay(frame_rgb, mask_small, out_w, out_h,
                                     query, info["best_score"])
            writer.write(out_bgr)

            if vis_every and frame_id % vis_every == 0:
                H_vis, W_vis = IMAGE_SIZE
                vis_in = cv2.resize(frame_bgr, (W_vis, H_vis))
                vis_out = cv2.resize(out_bgr, (W_vis, H_vis))
                sep = np.ones((H_vis, 6, 3), dtype=np.uint8) * 255
                canvas = np.concatenate([vis_in, sep, vis_out], axis=1)
                cv2.putText(canvas,
                            f"frame {frame_id}  |  "
                            f"yaw={info['yaw']:.1f} pitch={info['pitch']:.1f}  |  "
                            f"q: {query[:40]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
                vis_path = os.path.join(out_frames_dir,
                                        f"frame_{frame_id:06d}.jpg")
                cv2.imwrite(vis_path, canvas,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                vis_count += 1

            frame_id += 1
            if frame_id % 30 == 0:
                el = time.time() - t0
                eta = el / frame_id * (n_total - frame_id)
                print(f"    frame {frame_id:>5d}/{n_total}  "
                      f"elapsed={el:.0f}s  eta={eta:.0f}s  "
                      f"yaw={info['yaw']:7.2f} pitch={info['pitch']:7.2f}")
    finally:
        cap.release()
        writer.release()

    total_el = time.time() - t0
    print(f"\n  ✅ video: {out_video}  ({total_el:.0f}s, "
          f"{frame_id/max(1,total_el):.2f} fps)")
    if vis_count:
        print(f"  ✅ {vis_count} frames saved: {out_frames_dir}")
    print(f"\n  If you need audio:")
    print(f"    ffmpeg -i {out_video} -i {video_path} -c copy "
          f"-map 0:v:0 -map 1:a:0 {out_video.replace('.mp4', '_audio.mp4')}")


# ================= Command-line entry point =================

def main():
    parser = argparse.ArgumentParser(description="Segment a referred object in an ERP image or video.")
    parser.add_argument("--mode", choices=("image", "video"), default="image")
    parser.add_argument("--input", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--checkpoint-dir", default=OUTPUT_DIR,
                        help="Root containing stage0/best, stage2/best and ranker.pt")
    parser.add_argument("--out_dir", default=os.path.join(OUTPUT_DIR, "inference"))
    parser.add_argument("--name", help="Output basename, defaulting to the input stem")
    parser.add_argument("--vis_every", type=int, default=30,
                        help="Write a comparison every N video frames (0 disables)")
    parser.add_argument("--max_frames", type=int)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"input file does not exist: {args.input}")
    if not args.query.strip():
        parser.error("--query must contain a referring expression")

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    tag = _safe(args.query)

    print("=" * 70)
    print(f"OV-PRS  mode={args.mode}")
    print(f"  input  : {args.input}")
    print(f"  query  : {args.query}")
    print(f"  out_dir: {args.out_dir}")
    print(f"  name   : {name}")
    print(f"  PW in  : {PW_INPUT_W}x{PW_INPUT_H}")
    print(f"  Ranker : {FEAT_DIM}-dim")
    if args.mode == "video":
        print(f"  vis_every  : {args.vis_every}")
        print(f"  max_frames : {args.max_frames}")
    print("=" * 70)

    pipeline = Pipeline(args.checkpoint_dir)

    if args.mode == "video":
        out_video = os.path.join(args.out_dir, f"{name}_{tag}.mp4")
        out_frames = os.path.join(args.out_dir, f"{name}_{tag}_frames")
        process_video(pipeline, args.input, args.query, out_video, out_frames,
                      vis_every=args.vis_every, max_frames=args.max_frames)
    else:
        out_img = os.path.join(args.out_dir, f"{name}_{tag}.jpg")
        process_image(pipeline, args.input, args.query, out_img)


if __name__ == "__main__":
    main()
