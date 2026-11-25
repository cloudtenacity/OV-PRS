"""Load the PanoWorld-Hstar model from its upstream checkout."""

import importlib.util
import sys

import torch
from safetensors import safe_open

from config_panoworld import DEVICE, PANOWORLD_PATH, PANOWORLD_REPO, build_cfg
from prepare_panoworld_checkpoint import LEGACY_PREFIXES, weight_files


def _upstream_loader():
    train_dir = PANOWORLD_REPO / "train_copy_hstar"
    utils_file = train_dir / "utils.py"
    if not utils_file.is_file():
        raise FileNotFoundError(
            f"Could not find {utils_file}. Set PANOWORLD_REPO to the official PanoWorld checkout."
        )
    spec = importlib.util.spec_from_file_location("_ovprs_panoworld_utils", utils_file)
    module = importlib.util.module_from_spec(spec)
    search_path = sys.path[:]
    sys.path.insert(0, str(train_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = search_path
    return module


def _validate_checkpoint():
    if not (PANOWORLD_PATH / "config.json").is_file():
        raise FileNotFoundError(
            f"Set PANOWORLD_MODEL to a local PanoWorld-Hstar checkpoint. "
            f"Could not find config.json in {PANOWORLD_PATH}."
        )
    has_adapter = False
    for path in weight_files(PANOWORLD_PATH):
        with safe_open(path, framework="pt", device="cpu") as weights:
            keys = weights.keys()
        if any(key.startswith(LEGACY_PREFIXES) for key in keys):
            raise ValueError(
                "The PanoWorld-Hstar checkpoint uses legacy tensor names. "
                "Run prepare_panoworld_checkpoint.py, then update PANOWORLD_MODEL."
            )
        has_adapter |= any(key.startswith("erp_patch_adapter.") for key in keys)
    if not has_adapter:
        raise ValueError(f"No ERP patch adapter weights found in {PANOWORLD_PATH}.")


def load_panoworld():
    """Return the upstream configuration, processor, tokenizer, and frozen model."""
    if not torch.cuda.is_available():
        raise RuntimeError("PanoWorld-Hstar inference requires CUDA.")
    _validate_checkpoint()
    upstream = _upstream_loader()
    cfg = build_cfg()
    upstream.set_seed(cfg.training.seed)

    processor, tokenizer = upstream.load_processor_and_tokenizer(cfg)
    model = upstream.load_model(cfg)

    backbone_dtype = next(model.parameters()).dtype
    if hasattr(model, "erp_patch_adapter"):
        model.erp_patch_adapter = model.erp_patch_adapter.to(backbone_dtype)

    model.eval()
    model.requires_grad_(False)
    model.to(DEVICE)
    return cfg, processor, tokenizer, model
