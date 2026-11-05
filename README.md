# National flag identification in maritime-oriented imagery (ResNet18 + YOLOv8n)

Code for the manuscript *National Flag Identification in Maritime-Oriented Imagery Using ResNet18 and YOLOv8n: A Leakage-Controlled Proof-of-Concept Study* (V. V. Sattin, A. Duarte).

**This repository contains scripts only. No photograph or dataset is included** (the images come from third-party sources with their own terms).

## What the code does

| Script | Purpose |
|---|---|
| `01_build_groups.py` | Builds near-duplicate groups (MD5, 256-bit dHash with Hamming <= 24, 16x16 thumbnail correlation >= 0.90) per class and the 5 group-aware, class-stratified splits (`splits/`). |
| `02_run_internal_experiments.py` | Internal experiments (5 seeds): direct / central crop / YOLO crop x none / uniform / targeted augmentation; optional architecture baselines. Resumable. |
| `10_extract_frames.py`, `11_import_photos.py` | Prepare maritime frames from videos or collected photos. |
| `12_prepare_maritime.py` | Split of the maritime photographs by source, statistics, box/attribute templates. |
| `13_run_maritime_experiments.py` | Trains the single-class flag detector and classifiers A/B, evaluates localization (mAP@0.5 and mAP@0.5:0.95), end-to-end identification, negatives, subgroups and timing. |
| `fix_labels.py` | One-off clean-up of the exported box labels used in this study (duplicate lines, three missing boxes). |
| `check_overlap.py` | Compares the maritime photographs with the internal images (MD5, dHash, thumbnail correlation). |
| `print_env.py` | Prints Python/library versions and hardware, and a pinned requirements list. |
| `make_metadata_zip.py` | Packs non-image metadata (splits, labels, result tables) into a zip. |
| `common.py` | Shared code (data loading, transforms, training, detection, metrics, timing). |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python print_env.py
```

Experiments in the paper ran on a laptop CPU (Intel Core i5-1235U, 20 GB RAM, no CUDA GPU; Python 3.11) under Windows.
Pretrained weights (ResNet18 etc. and `yolov8n.pt`) are downloaded automatically on first use.

## Data layout expected (you must supply the images)

```
dataset/<Class>/<image>.png|jpg      # internal images, 25 classes
maritime/frames/<name>.jpg           # maritime photographs (or extracted frames)
maritime/labels/<name>.txt           # YOLO-format boxes (class index follows maritime/classes.txt); empty/absent = negative
```

Classes: Argentina, Australia, Belgium, Brazil, Canada, Chile, China, Denmark, France, Germany, Greece, India, Italy, Japan, Mexico, Netherlands, Norway, Portugal, Russia, South Korea, Spain, Sweden, Turkey, UK, USA.
Box annotation was done in CVAT (labels in `maritime/cvat_labels.json`, export "YOLO 1.1").

## Run

```bash
python 01_build_groups.py
python 02_run_internal_experiments.py --seeds 0 1 2 3 4
# architecture baselines
python 02_run_internal_experiments.py --seeds 0 1 2 3 4 --configs direct_targeted_resnet50 direct_targeted_mobilenet_v3_large direct_targeted_efficientnet_b0
# maritime part (needs your own annotated photographs)
python 12_prepare_maritime.py --unit image --fractions 0.5 0.1 0.4
python 13_run_maritime_experiments.py --det-epochs 30 --det-imgsz 512
python check_overlap.py
```

Outputs go to `results/` (tables as `.md` / `.csv`, per-run `.json`, confusion matrices).
Run the scripts from the repository root, since paths are relative to the script folder.

## Notes

- Augmentation is online and applied to the training subset only; the test subset is never used for training, model selection or thresholds.
- Source terms: images from Kaggle, Wikimedia Commons, Ships in Motion (YouTube), ShipSpotting and Getty Images remain subject to their original terms and are not redistributed.
- Licence: add a `LICENSE` file of your choice before publishing.
