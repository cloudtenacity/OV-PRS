"""Paths and PanoWorld-Hstar inference settings for OV-PRS."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _path(name, default):
    return Path(os.environ.get(name, default)).expanduser().resolve()


DATA_ROOT = _path("OVPRS_DATA_ROOT", PROJECT_ROOT / "data")
CACHE_DIR = _path("OVPRS_CACHE_DIR", PROJECT_ROOT / "cache")
OUTPUT_DIR = _path("OVPRS_OUTPUT_DIR", PROJECT_ROOT / "outputs")
PANOWORLD_REPO = _path("PANOWORLD_REPO", PROJECT_ROOT / "third_party" / "PanoWorld")
PANOWORLD_PATH = _path("PANOWORLD_MODEL", PROJECT_ROOT / "checkpoints" / "PanoWorld_Hstar")
MASK2FORMER_PATH = os.environ.get("OVPRS_MASK2FORMER", "facebook/mask2former-swin-base-coco-panoptic")
CLIP_PATH = os.environ.get("OVPRS_CLIP", "openai/clip-vit-large-patch14")
BENCHMARK_SPLITS = ("base_train", "base_test", "novel_test", "ood")

# ==================== Generation settings ====================
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
NUM_BEAMS = 1
SEED = 42
DEVICE = "cuda"
ATTN_IMPL = "sdpa"

# ==================== Image preprocessing ====================
MIN_PIXELS = 16384
MAX_PIXELS = 2097160
SHORTEST_EDGE = 16384
LONGEST_EDGE = 2097160

# ==================== ERP Adapter ====================
ERP_ENABLED = True
ERP_POS_MODE = "paper"
ERP_STAGE = "patch"
ERP_TARGET = "pooler"
ERP_ADAPTER_TYPE = "da2_cross_attn"
ERP_EMBED_TYPE = "fourier"
ERP_HIDDEN_DIM = 512
ERP_NUM_HEADS = 8
ERP_GATE_INIT = 0.01
ERP_USE_LAYERNORM = True

IMAGE_TOKEN = "<image>"
MODEL_MAX_LENGTH = 163840


def build_cfg():
    """Build the configuration consumed by the upstream PanoWorld loader."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "model": {
            "name_or_path": str(PANOWORLD_PATH),
            "trust_remote_code": False,
            "torch_dtype": "bfloat16",
            "attn_implementation": ATTN_IMPL,
            "cache_dir": None,
            "image_token": IMAGE_TOKEN,
            "model_max_length": MODEL_MAX_LENGTH,
            "trainable_modules": {"all": True},
        },
        "data": {
            "train_jsonl": None,
            "eval_jsonl": None,
            "system_prompt_path": None,
            "image_root": None,
            "train_max_samples": None,
            "eval_max_samples": None,
            "shuffle": False,
            "eval_shuffle": False,
            "prompt_format": "chat_template",
            "system_prompt": None,
            "auto_insert_media_placeholders": True,
            "strict_image_checks": True,
            "skip_missing_images": True,
            "eval_method": "generation",
            "eval_metric": "choice_accuracy",
            "eval_use_generation": True,
            "eval_disable_thinking": True,
            "eval_print_predictions": True,
            "eval_generation_max_new_tokens": MAX_NEW_TOKENS,
            "eval_generation_do_sample": DO_SAMPLE,
            "eval_generation_num_beams": NUM_BEAMS,
            "image_processor": {
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "size": {
                    "shortest_edge": SHORTEST_EDGE,
                    "longest_edge": LONGEST_EDGE,
                },
            },
        },
        "erp": {
            "enabled": ERP_ENABLED,
            "pos_mode": ERP_POS_MODE,
            "stage": ERP_STAGE,
            "target": ERP_TARGET,
            "adapter_type": ERP_ADAPTER_TYPE,
            "cross_attn_embed_type": ERP_EMBED_TYPE,
            "hidden_dim": ERP_HIDDEN_DIM,
            "num_heads": ERP_NUM_HEADS,
            "gate_init": ERP_GATE_INIT,
            "use_layernorm": ERP_USE_LAYERNORM,
        },
        "training": {
            "output_dir": "./outputs/infer_tmp",
            "seed": SEED,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-6,
            "weight_decay": 0.01,
            "warmup_steps": 0.0,
            "lr_scheduler_type": "cosine",
            "optim": "adamw_torch",
            "num_train_epochs": 1,
            "fp16": False,
            "bf16": True,
            "logging_steps": 5,
            "save_steps": 25,
            "max_steps": -1,
            "save_strategy": "steps",
            "max_shard_size": "5GB",
            "save_total_limit": 2,
            "report_to": [],
            "run_name": "infer",
            "per_device_eval_batch_size": 1,
            "eval_steps": 25,
            "eval_strategy": "no",
            "load_best_model_at_end": False,
            "metric_for_best_model": None,
            "greater_is_better": None,
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "gradient_checkpointing": False,
            "max_grad_norm": 1.0,
            "deepspeed": None,
        },
        "run": {
            "do_train": False,
            "do_eval": False,
            "resume_from_checkpoint": False,
            "eval_before_train": False,
        },
        "wandb": {
            "project": "pano",
            "entity": None,
            "name": "infer",
            "tags": [],
            "mode": "offline",
            "group": None,
            "notes": "",
        },
    })
