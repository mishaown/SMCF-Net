# SMCF-Net: Bi-temporal SAR Flood Segmentation

SMCF-Net segments flooded areas from paired pre-flood and post-flood synthetic aperture radar (SAR) imagery. It combines a shared encoder, cross-temporal attention, ConvLSTM temporal processing, multi-scale difference aggregation, and a progressive upsampling decoder.

This repository provides the model, training utilities, configuration, and fixed train/validation/test splits for **S1GFloods** and **OMBRIA-S1**. Raw imagery, masks, pretrained/trained checkpoints, and experiment results are obtained or generated separately.

## Installation

Use **Python 3.10–3.12**. Install a CUDA-enabled PyTorch environment suitable for your GPU, then run from the repository root:

```bash
python -m pip install -e .
```

Dependencies and supported version ranges are declared in [pyproject.toml](pyproject.toml).

The default encoder is `timm`'s `efficientnetv2_rw_t`. With `pretrained: true` in [configs/study.yaml](configs/study.yaml), encoder weights are downloaded on first use. For offline use, cache these weights beforehand or set `pretrained: false`; random initialization changes the experimental setup.

## Data preparation

Obtain the datasets separately under their respective distribution terms. The [split CSVs](splits/) specify the exact filenames and relative paths the loader expects. Preserve the filenames and split membership.

Place prepared imagery and masks in the following layout:

```text
data/datasets/
├── s1gfloods/
│   ├── A/image_466.png
│   ├── B/image_466.png
│   └── Label/image_466.png
└── ombrias1/
    └── train/
        ├── BEFORE/S1_before_0545.png
        ├── AFTER/S1_after_0545.png
        └── MASK/S1_mask_0545.png
```

These are example paths. Every file named in the selected dataset's three split CSVs must be present. OMBRIA-S1 paths use the original `train`/`test` folders; the CSVs define this study's partitions independently of those folder names.

For externally stored data, pass `--data-root /path/to/data`, where that directory contains `datasets/`. The `SMCF_NET_DATA_ROOT` environment variable provides the same override. Data is excluded from Git, and each training command requires only its selected dataset.

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| S1GFloods | 4,288 | 536 | 536 |
| OMBRIA-S1 | 555 | 69 | 70 |

Sample IDs are disjoint across splits; this alone does not establish geographic or event independence.

## Training

Train the full SMCF-Net model from the repository root:

```bash
python -m benchmark.train --dataset s1gfloods --seed 23534
python -m benchmark.train --dataset ombrias1 --seed 23534
```

With data stored elsewhere:

```bash
python -m benchmark.train --dataset s1gfloods --data-root /path/to/data
```

The default variant, `parallel`, is the full model. Train the primary ablations independently using the same config and seed:

```bash
python -m benchmark.train --dataset s1gfloods --seed 23534 --variant no_temporal
python -m benchmark.train --dataset s1gfloods --seed 23534 --variant attention_only
python -m benchmark.train --dataset s1gfloods --seed 23534 --variant memory_only
```

Use `--help` to see additional SMCF-Net variants and options.

### Default configuration

Settings are provided in [configs/study.yaml](configs/study.yaml):

| Setting | Value |
| --- | --- |
| Input size | Paired 256 × 256 images |
| Batch size | 16 |
| Optimizer | Adam |
| Training budget | 40,000 updates |
| Learning rate | 0.0005 |
| Warm-up | 200 updates |
| Learning-rate schedule | Polynomial decay |
| Loss | Tversky, α = 0.3, β = 0.7 |
| Seed | 23534 |

Training applies paired crop/resize and horizontal/vertical flips. Temporal exchange is disabled. The data loader preserves the normalization and mask handling used by the model.

Use `--config path/to/config.yaml` to change settings. On Windows or constrained machines, set `data.num_workers` to `0` if needed. `--device cpu` supports small checks; full training is intended for a CUDA GPU.

### Outputs and checkpoint selection

Outputs default to:

```text
results/study/<dataset>/smcf_net-<variant>-seed<seed>/
```

Use `--output path/to/new/run` for a different location. Existing output folders are refused; training does not resume a partial run.

Outputs include `best.pt`, `last.pt`, configuration, source/split hashes, package versions, training history in JSON and CSV, and training time/memory metadata. Generated outputs are ignored by Git.

Checkpoint selection uses global validation F1 at threshold `0.5` after each epoch; ties select the later checkpoint. Training never evaluates test data.

## Checkpoint inference

A training checkpoint includes model state and configuration. Load a checkpoint you trained locally:

```python
import torch
from benchmark.train import construct

checkpoint = torch.load("path/to/best.pt", map_location="cpu", weights_only=True)
model = construct(checkpoint["config"], pretrained=False)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
```

Use `benchmark.data.ManifestDataset` and `PairedTransform(train=False)` for the same preprocessing. `model(pre, post)["logits"]` has shape `[B, 2, H, W]`; softmax channel `1` gives flood probability.

```python
# pre and post are preprocessed tensors of shape [B, 3, H, W].
prediction = model.predict(pre, post, threshold=0.5)
probabilities = prediction["probabilities"]
binary_masks = prediction["predictions"]
```

Select any alternative threshold using validation data only, and freeze it before evaluating the held-out test split.

## Repository layout

```text
SMCF-Net/
├── models/          # SMCF-Net, its helpers, and baseline model source files
├── benchmark/       # Data loading, training, losses, metrics, and history
├── configs/         # Reproduction settings
├── splits/          # Six original fixed split CSVs
├── LICENSES/        # Source license
├── pyproject.toml
├── requirements.txt
└── README.md
```

Nine baseline model `.py` files are included as source references with their existing attributions preserved. Supporting baseline backbones, registry entries, training workflows, and checkpoints are excluded. Baseline files that import omitted backbones require those dependencies to run. The public training command and registry remain specific to SMCF-Net.

## License

See [LICENSES/SMCF-Net-MIT.txt](LICENSES/SMCF-Net-MIT.txt). Preserve existing source attributions and applicable upstream terms for third-party code.
