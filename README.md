# National flag identification in maritime-oriented imagery (ResNet18 + YOLOv8n)

Code for the manuscript *National Flag Identification in Maritime-Oriented Imagery Using ResNet18 and YOLOv8n: A Proof-of-Concept Study* (V. V. Sattin, A. Duarte).

**This repository contains scripts only. No photograph or dataset is included** (the images come from third-party sources with their own terms).

## What the code does

| Script | Purpose |
|---|---|
| `01_build_groups.py` | Builds near-duplicate groups (MD5, 256-bit dHash with Hamming <= 24, 16x16 thumbnail correlation >= 0.90) per class and the 5 group-aware, class-stratified splits (`splits/`). |
| `02_run_internal_experiments.py` | Internal experiments (5 seeds): direct / central crop / YOLO crop x none / uniform / targeted augmentation; optional architecture baselines. Resumable. |
| `10_extract_frames.py`, `11_import_photos.py` | Prepare maritime frames from videos or collected photos. |
| `12_prepare_maritime.py` | Split of the maritime photographs by source, statistics, box/attribute templates. |
| `13_run_maritime_experiments.py` | Trains the single-class flag detector and classifiers A/B, evaluates localization (mAP@0.5 and mAP@0.5:0.95), end-to-end identification, negatives, subgroups and timing. |
| `14_compare_classifiers_maritime.py` | Compares ResNet18 and MobileNetV3-Large on the same maritime test detections from the fixed detector (accuracy on annotated boxes, end-to-end metrics, paired differences with bootstrap intervals, classifier-only and full-pipeline latency). Reuses the detector and classifier B of script 13; trains an equivalent MobileNetV3-Large classifier. |
| `15_compare_single_stage.py` | Compares the two-stage pipeline (single-class detector + ResNet18) with a single-stage YOLOv8n trained on the same maritime photographs to predict flag location and country: end-to-end performance (correct localization and country), frame-level accuracy, false alarms on negative frames, paired differences and time per frame. |
| `16_eval_negatives.py` | False alarms on extra flag-free photographs for the two-stage pipeline and the single-stage models at confidence thresholds 0.1/0.25/0.5/0.7, with Wilson 95% intervals (`--neg-dir`). |
| `17_rejection_by_probability.py` | Exploratory effect of requiring a minimum classifier probability (classifier B) on end-to-end precision/recall (maritime test) and on false alarms (test negatives plus extra negatives). Thresholds are not tuned on independent data. |
| `18_recall_by_size.py` | Detection and end-to-end recall of the maritime test boxes by flag size (512-px scale), and a simulated-distance analysis (photographs shrunk onto a grey canvas, pooled by effective size). |
| `19_eval_unknown_flags.py` | Flags outside the 25 classes: images accepted by the two-stage pipeline (with and without a probability filter) and by the single-stage models. Expects `<root>/A` (national flags of other countries) and `<root>/B` (other flags); file names containing the keywords of `NAVAL25` in the script (naval ensigns of countries among the 25) are analysed apart, so adapt that list to your own file names. |
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

**Use the pinned versions.** The detection results (number of boxes kept after non-maximum suppression) differ between `ultralytics` versions: with 8.4.216 the maritime detector gave fewer extra boxes than with the pinned 8.4.202 used for all reported results. Install `ultralytics==8.4.202`. Reading `.avif` images (scripts 16-19) additionally needs the optional package `pillow-avif-plugin`.

**Trained weights are not committed** (`*.pt` is ignored). Scripts 14-19 reuse the weights written by scripts 13 and 15 (`work/maritime_detector_seed0/`, `work/maritime_single_stage_seed0_e30|e100/`, `results/maritime/classifier_B.pt`); run those first, or copy the weights from the archived release if provided.

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
python 14_compare_classifiers_maritime.py
python 15_compare_single_stage.py
# extra analyses (need the extra images described below)
python 16_eval_negatives.py --neg-dir "<folder with flag-free photographs>"
python 17_rejection_by_probability.py --neg-dir "<folder with flag-free photographs>"
python 18_recall_by_size.py
python 19_eval_unknown_flags.py --root "<folder with subfolders A and B>"
python check_overlap.py
```

`13_run_maritime_experiments.py` was run with `--det-epochs 30 --det-imgsz 512` (seed 0; all other options at their defaults) to produce the maritime tables of the paper; the arguments are stored in `results/maritime/maritime_results.json`. Classifiers A and B are ResNet18 models trained with the *targeted* augmentation policy (five weak classes, repeated three times per epoch). Both use the training and test subsets of the seed-0 internal split as training data and its validation subset for early stopping, because the maritime test photographs are the evaluation set of that part; B adds the crops of the maritime training boxes (and of the validation boxes to the validation set).

The extra photographs used by scripts 16, 17 and 19 (96 flag-free photographs; 30 national flags of other countries, 14 other flags and 7 naval ensigns of countries among the 25 classes) are not included; their results are in `results/maritime/` (`negatives_eval`, `rejection_by_probability`, `recall_by_size`, `unknown_flags`, `compare_classifiers`, `single_stage_comparison`).

Outputs go to `results/` (tables as `.md` / `.csv`, per-run `.json`, confusion matrices).
Run the scripts from the repository root, since paths are relative to the script folder.

## Notes

- Augmentation is online and applied to the training subset only; the test subset is never used for training, model selection or thresholds.
- Source terms: images from Kaggle, Wikimedia Commons, Ships in Motion (YouTube), ShipSpotting and Getty Images remain subject to their original terms and are not redistributed.
- Licence: MIT (see `LICENSE`); it covers the code only, not the images.
