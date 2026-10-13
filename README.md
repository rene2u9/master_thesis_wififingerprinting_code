# WiFi Fingerprinting in Warehouse Environments

Code accompanying the master's thesis *Indoor-Lokalisierung mittels
WiFi-Fingerprinting in Lagerumgebungen* (Universität Leipzig).

The framework compares 14 deep-learning backbones from five architecture
families (CNN, Transformer, state-space models, graph networks, MLP/autoencoder)
plus a WKNN baseline for predicting floor coordinates from WiFi RSSI scans.
Each of the 14 backbones exists as a regression model and as a classification
twin that classifies over storage-bay centroids and reads out a position by
top-k soft-argmax. ResNet-18 is included as an additional regression model.

## Dataset

The dataset (700,512 fingerprints) is attached to the
[v1.0 release](https://github.com/rene2u9/master_thesis_wififingerprinting_code/releases/tag/v1.0):

- [`wifi_fingerprints_anon.db.gz`](https://github.com/rene2u9/master_thesis_wififingerprinting_code/releases/download/v1.0/wifi_fingerprints_anon.db.gz) (235 MB, 2.9 GB unpacked)
- [`wifi_fingerprints_anon.db.gz.sha256`](https://github.com/rene2u9/master_thesis_wififingerprinting_code/releases/download/v1.0/wifi_fingerprints_anon.db.gz.sha256)

It is pseudonymised: network names are replaced by `warehouse*`, the access
points by locally administered MAC addresses (`02:00:00:00:XX:YY`, the last
octet kept so that the grouping into physical APs is unchanged) and the
reading hashes by random tokens. Storage places, floor coordinates and
timestamps are unchanged, so the splits and results of the regimes R1 and R2
reproduce. The trajectory experiments of regime R3 need device and
scan-capture fields that the extract does not contain.

```bash
mkdir -p data
gh release download v1.0 --repo rene2u9/master_thesis_wififingerprinting_code
shasum -a 256 -c wifi_fingerprints_anon.db.gz.sha256
gunzip -c wifi_fingerprints_anon.db.gz > data/wifi_fingerprints.db
```

## Setup

Tested with Python 3.13 and PyTorch 2.11 (CUDA).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Spatial-Mamba and MambaVision require a CUDA GPU. Spatial-Mamba additionally
needs the custom CUDA kernels from the
[Spatial-Mamba repository](https://github.com/EdwardChasel/Spatial-Mamba).
All other models also run on CPU. Pretrained weights are not included:
place `vim_s_midclstok_80p5acc.pth` from [Vim](https://github.com/hustvl/Vim)
in `models/vision_mamba/` and the Spatial-Mamba-T checkpoint as
`models/spatial_mamba/spatial_mamba_t.pth`. The remaining ImageNet weights are
loaded through `timm`.

## Data format

`data/wifi_fingerprints.db` (path set in `config.yaml`, `data.db_path`). Besides
`fingerprint_hash` (deduplication), `ap_positions` (reconstructed AP positions)
and `pickspots` (bay coordinates), the main table is:

```sql
CREATE TABLE wifi_fingerprints (
    id INTEGER PRIMARY KEY,
    storage_place_desc TEXT NOT NULL,  -- "aisle;rack;bay;level"
    latitude REAL NOT NULL,            -- floor coordinates in metres, not GPS
    longitude REAL NOT NULL,
    created_at TEXT NOT NULL,          -- ISO timestamp
    readings TEXT NOT NULL             -- JSON: [{"ssid", "bssid", "level"}, ...]
);
```

Conventions: a missing access point is -100 dBm, RSSI is min-max normalised
from [-90, -50] dBm to [0, 1], and the access-point order is fixed per run
(`sorted_aps.json` in the experiment directory). Set `data.filter_ssid_prefix`
in `config.yaml` to the SSID prefix of the warehouse network to drop foreign
networks. `scripts/export_temporal.py` documents how the data was exported
from the production PostgreSQL database with labels taken from the PickingSpots
table. It needs `psycopg2` and access to that database and is not required
for working with the released extract.

## Evaluation regimes

- R1: random draw of 100,000 deduplicated rows, random 70/15/15 split (main
  benchmark, all 14 backbones).
- R2: the same rows with a temporal or a bay-wise split (leakage control).
- R3: timestamped capture of real picking sessions with a temporal 85/15 split
  (trajectory filtering, `scripts/sub1m/lab.py`).

## Usage

```bash
python train.py --models transformer_cls_100k        # one named run from config.yaml
python train.py --models ffn_cls_100k --fast-dev-run # quick smoke test
python train.py                                      # all runs in config.yaml
tensorboard --logdir experiments/
```

All hyperparameters live in `config.yaml`. The `runs:` section defines named
configurations (backbone, head, input representation, training recipe).
Scripts in `scripts/`:

```bash
python scripts/wknn_baseline.py --k 9                 # WKNN baseline, R1
python scripts/wknn_clean.py                          # leave-one-out label cleaning
python scripts/centroid_baseline.py --run resnet50_cls100k
python scripts/eval_cls_readout.py --ckpt experiments/resnet50_cls100k
python scripts/eval_ap_dropout.py --model wknn --run resnet50_cls100k
python scripts/deduplicate_fingerprints.py            # rebuild the dedup hash table
python scripts/audit_coordinate_consistency.py data/wifi_fingerprints.db
```

The trajectory experiments (HMM filtering, reject option) are in
`scripts/sub1m/lab.py`.

## Structure

```
src/data/       dataset, deduplication, normalisation, image and graph encodings
src/models/     backbones, regression and classification heads, model registry
src/training/   Lightning trainer, optimisers, checkpoint utilities
src/metrics/    mean/median error and CDF metrics
scripts/        baselines (WKNN, centroid), evaluations, deduplication, export, audit
train.py        entry point for training runs
config.yaml     runs and hyperparameters
```

## Models

| Family | Backbones |
|---|---|
| CNN | ResNet-50 (plus ResNet-18), EfficientNet-B0, DSCNN (after DS-CNN, Zhu et al. 2023) |
| Transformer | per-AP Transformer, ViT, Swin-T |
| State space | Vision Mamba, Spatial-Mamba, MambaVision, custom Mamba |
| Graph | GCN, VGAE |
| MLP / autoencoder | FFN, SAE |

Parameter counts, input representations and the full run configurations are
listed in Appendix A of the thesis.
