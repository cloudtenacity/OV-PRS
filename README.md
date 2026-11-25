# OV-PRS

Open-vocabulary referring segmentation for 360° ERP panoramas.

**Release status.** The OV-PRS trained checkpoints and OV-PanoReferSeg dataset are pending release.

## Install

Use Linux with an NVIDIA GPU. Run the commands from the repository root.

```bash
conda env create -f environment.yml
conda activate ov-prs
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Transformers downloads the initial Mask2Former and CLIP weights on first use. For PanoWorld-Hstar, follow the upstream setup and point to its checkout and downloaded checkpoint. [PanoWorld paper](https://arxiv.org/abs/2605.13169) · [PanoWorld setup](https://github.com/wcpcp/PanoWorld#environment-setup)

```bash
export PANOWORLD_REPO=/path/to/PanoWorld
export PANOWORLD_MODEL=/path/to/PanoWorld_Hstar
```

If the loader reports legacy tensor names, convert the Hstar checkpoint and update its path.

```bash
python prepare_panoworld_checkpoint.py "$PANOWORLD_MODEL" checkpoints/PanoWorld_Hstar_fixed
export PANOWORLD_MODEL="$PWD/checkpoints/PanoWorld_Hstar_fixed"
```

## Train and evaluate

Put the benchmark in `data/` or set `OVPRS_DATA_ROOT`. Repeat this layout for `base_train`, `base_test`, `novel_test`, and `ood`.

```text
data/<split>/annotations.csv
data/<split>/data/<category>/original/<unique_name>.jpg
data/<split>/data/<category>/mask/<unique_name>.png
```

The CSV has `unique_name`, `category`, `description`, and `target_word` columns.

```bash
python cash_panoworld.py
TRAIN_STAGE=0 python trainv1_stage1_2.py
TRAIN_STAGE=1 python trainv1_stage1_2.py
TRAIN_STAGE=2 python trainv1_stage1_2.py
python trainv1_stage3.py --stage all
python evaluate.py --no_vis
```

Training writes checkpoints to `outputs/` by default. If using an existing OV-PRS checkpoint, pass its root to `evaluate.py` with `--checkpoint-dir`.

## Inference

Use the checkpoints produced by training or point `--checkpoint-dir` to your own OV-PRS checkpoint directory.

```bash
python inference_pipeline.py \
  --input /path/to/panorama.jpg \
  --query "the red fire extinguisher" \
  --checkpoint-dir outputs
```

For video input, use `--mode video` and pass an ERP video to `--input`.
