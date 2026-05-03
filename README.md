# StegaStamp Full Reproduction

This repository is organized for paper-style StegaStamp reproduction, not demo-only usage.

Implemented components:
- Encoder/decoder/critic training pipeline with paper-inspired schedule.
- Perturbation ablation profiles: `none`, `pixelwise`, `spatial`, `all`.
- BiSeNet-style detector training for StegaStamp region segmentation.
- Quadrilateral fitting and homography rectification to `400x400`.
- BCH integration path for 56-bit payload inside 100-bit model message budget.
- Synthetic ablation and controlled pipeline evaluation scripts with percentile outputs.

## Repository Structure

- `stegastamp/models/encoder_decoder.py`: encoder + STN decoder.
- `stegastamp/models/critic.py`: critic network.
- `stegastamp/models/detector_bisenet.py`: detector model.
- `stegastamp/perturbations.py`: differentiable corruption module.
- `stegastamp/losses.py`: residual/LPIPS/critic losses.
- `stegastamp/data/mirflickr.py`: MIRFLICKR loader.
- `stegastamp/data/div2k_detector.py`: detector training sample synthesis.
- `stegastamp/geometry/homography.py`: mask -> quad -> rectified crop.
- `stegastamp/ecc/bch.py`: BCH encode/decode wrapper.
- `scripts/train_encoder_decoder.py`: main encoder/decoder/critic training.
- `scripts/train_detector.py`: detector training.
- `scripts/eval_synthetic_ablation.py`: perturbation robustness evaluation.
- `scripts/eval_decode_pipeline.py`: full detector->rectify->decode evaluation.

## Install

```bash
py -3 -m pip install -r requirements.txt
```

## Data Layout

### 1) MIRFLICKR-like training root (ImageFolder format)

```text
mirflickr_root/
  images/
    0001.jpg
    0002.jpg
```

### 2) Detector data

```text
div2k_root/
  *.png
stamps_root/
  *.png
```

`stamps_root` should contain encoded StegaStamp crops used for detector compositing.

## Reproduction Commands

### A) Train encoder/decoder/critic

Full perturbation model (paper main variant):

```bash
py -3 scripts/train_encoder_decoder.py ^
  --data-dir mirflickr_root ^
  --output-dir runs/encdec_all ^
  --perturbation-profile all ^
  --use-bch
```

Ablation variants:

```bash
py -3 scripts/train_encoder_decoder.py --data-dir mirflickr_root --output-dir runs/encdec_none --perturbation-profile none
py -3 scripts/train_encoder_decoder.py --data-dir mirflickr_root --output-dir runs/encdec_pixelwise --perturbation-profile pixelwise
py -3 scripts/train_encoder_decoder.py --data-dir mirflickr_root --output-dir runs/encdec_spatial --perturbation-profile spatial
```

### B) Train detector

```bash
py -3 scripts/train_detector.py ^
  --backgrounds-dir div2k_root ^
  --stamps-dir stamps_root ^
  --output-dir runs/detector
```

### C) Synthetic ablation evaluation

```bash
py -3 scripts/eval_synthetic_ablation.py ^
  --checkpoint runs/encdec_all/checkpoint_epoch_029.pt ^
  --data-dir mirflickr_root ^
  --output-dir runs/eval_synth_all
```

Output: `synthetic_ablation.json` with mean and percentile bit-accuracy by perturbation and strength.

### D) Full decode pipeline evaluation

```bash
py -3 scripts/eval_decode_pipeline.py ^
  --decoder-checkpoint runs/encdec_all/checkpoint_epoch_029.pt ^
  --detector-checkpoint runs/detector/detector_epoch_019.pt ^
  --images-dir captured_images ^
  --labels-json labels.json ^
  --capture-metadata-csv capture_metadata.csv ^
  --output-dir runs/eval_pipeline ^
  --use-bch
```

Expected metadata CSV columns:
- `image_name`
- `camera`
- `media`

Expected labels JSON format:

```json
{
  "img_0001.jpg": "010101...100bits...",
  "img_0002.jpg": "110100...100bits..."
}
```

## Reproducibility Notes

- Training checkpoints include config, epoch, perturbation profile, and git hash.
- Use fixed seeds (`--seed`) for repeatability.
- For strict table replication, run all four perturbation profiles and report:
  - synthetic: mean + percentile bands by perturbation strength
  - controlled: 5th/25th/50th/mean bit accuracy per camera/media group

## Fidelity Notes

The implementation follows paper-described architecture and pipeline flow closely. Exact numeric parity with the original internal codebase may still differ due to unpublished hyperparameter details and hardware/data capture differences.
