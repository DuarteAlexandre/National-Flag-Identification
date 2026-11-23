"""R2.1: false alarms on extra negative images (no flag visible) for the two-stage pipeline and the
single-stage YOLOv8n models. Run from the repository root:
  python 16_eval_negatives.py --neg-dir <folder with negative images> [--seed 0]
Uses the weights already trained by scripts 13 and 15 (work/ and results/maritime/)."""
from __future__ import annotations
import argparse, json, sys
from math import sqrt
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_model, classify_pil_batch, crop_box, yolo_detect, pick_device, list_dataset
try:
    import pillow_avif  # noqa: F401  (optional, for .avif files)
except Exception:
    pass
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
EXT = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".avif", ".bmp"}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.7])
    ap.add_argument("--epochs-list", type=int, nargs="+", default=[30, 100])
    ap.add_argument("--out", default="results/maritime/negatives_eval")
    a = ap.parse_args()
    device = pick_device()
    _, classes = list_dataset()
    work = ROOT / "work"
    det = YOLO(str(work / f"maritime_detector_seed{a.seed}/runs/train/weights/best.pt"))
    clf = build_model("resnet18", len(classes), pretrained=False)
    clf.load_state_dict(torch.load(ROOT / "results/maritime/classifier_B.pt", map_location="cpu"))
    clf = clf.to(device).eval()
    singles = {f"single_e{e}": YOLO(str(work / f"maritime_single_stage_seed{a.seed}_e{e}/runs/train/weights/best.pt")) for e in a.epochs_list}
    files = sorted(p for p in Path(a.neg_dir).iterdir() if p.suffix.lower() in EXT)
    cmin = min(a.thresholds)
    rows = []
    for p in files:
        im = Image.open(p).convert("RGB")
        r = {"file": p.name, "size": list(im.size)}
        b, c = yolo_detect(det, im, a.imgsz, cmin, device)
        crops = [crop_box(im, x, a.margin) or im for x in b]
        pr = classify_pil_batch(clf, crops, device) if crops else np.zeros((0, len(classes)))
        r["two_stage"] = [{"box": [float(v) for v in bb], "conf": float(cc), "cls": classes[int(q.argmax())]} for bb, cc, q in zip(b, c, pr)]
        for tag, m in singles.items():
            res = m.predict(source=im, imgsz=a.imgsz, conf=cmin, device=device, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                r[tag] = []
            else:
                r[tag] = [{"box": [float(v) for v in x], "conf": float(y), "cls": classes[int(k)]}
                          for x, y, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int))]
        rows.append(r)
    n = len(rows)
    summary, md = {}, [f"# False alarms on {n} negative images\n", "| model | conf | images with >=1 detection | detections | rate | Wilson 95% CI |", "|---|---|---|---|---|---|"]
    for key in ["two_stage"] + list(singles):
        for t in a.thresholds:
            k = sum(any(d["conf"] >= t for d in r[key]) for r in rows)
            nd = sum(sum(d["conf"] >= t for d in r[key]) for r in rows)
            lo, hi = wilson(k, n)
            summary[f"{key}@{t}"] = {"images_with_false_alarm": k, "detections": nd, "n": n, "rate": k / n, "wilson95": [lo, hi]}
            md.append(f"| {key} | {t} | {k}/{n} | {nd} | {k/n:.3f} | [{lo:.3f}, {hi:.3f}] |")
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "per_image": rows}, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
