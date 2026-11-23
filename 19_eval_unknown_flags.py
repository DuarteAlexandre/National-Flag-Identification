"""R1.3: flags outside the 25 target classes. Folder layout: <root>/A (national flags of other countries),
<root>/B (other flags: naval ensigns, signal, pirate, miscellaneous). No annotation is needed: every image shows one
flag that is NOT a civil national flag of the 25 classes (naval ensigns of in-class countries are reported apart,
see NAVAL25). An image counts as 'accepted' when at least one detection survives the detector confidence (td) and
the minimum classifier probability (tc); the predicted country is then, by construction, one of the 25 classes.
Single-stage models have an extra 'other' class: a detection of class 'other' is a correct rejection.
Run from the repository root:  python 19_eval_unknown_flags.py --root "<Novos dados 2/R1.3>" """
from __future__ import annotations
import argparse, json, sys
from math import sqrt
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import build_model, classify_pil_batch, crop_box, yolo_detect, pick_device, list_dataset
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass
from ultralytics import YOLO
EXT = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".avif", ".bmp"}
NAVAL25 = ("marines usa", "marinha brasil", "marinha china", "mainha russia", "marinha holanda", "navy usa", "marinha noruega")

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--det-conf", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--clf-prob", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.8, 0.9])
    ap.add_argument("--out", default="results/maritime/unknown_flags")
    a = ap.parse_args()
    device = pick_device(); _, classes = list_dataset()
    names = [l.strip() for l in (ROOT / "maritime/classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    det = YOLO(str(ROOT / f"work/maritime_detector_seed{a.seed}/runs/train/weights/best.pt"))
    clf = build_model("resnet18", len(classes), pretrained=False)
    clf.load_state_dict(torch.load(ROOT / "results/maritime/classifier_B.pt", map_location="cpu")); clf = clf.to(device).eval()
    singles = {f"single_e{e}": YOLO(str(ROOT / f"work/maritime_single_stage_seed{a.seed}_e{e}/runs/train/weights/best.pt")) for e in (30, 100)}
    groups = {"A": [], "B": [], "NAVAL25": []}
    for sub in ("A", "B"):
        for p in sorted((Path(a.root) / sub).iterdir()):
            if p.suffix.lower() in EXT:
                g = "NAVAL25" if sub == "B" and any(k in p.stem.lower() for k in NAVAL25) else sub
                groups[g].append(p)
    per = {}
    for g, files in groups.items():
        for p in files:
            im = Image.open(p).convert("RGB")
            b, c = yolo_detect(det, im, 512, min(a.det_conf), device)
            cr = [crop_box(im, x, a.margin) or im for x in b]
            pr = classify_pil_batch(clf, cr, device) if cr else np.zeros((0, len(classes)))
            r = {"group": g, "file": p.name, "two_stage": [{"conf": float(cc), "prob": float(q.max()), "cls": classes[int(q.argmax())]} for cc, q in zip(c, pr)]}
            for tag, m in singles.items():
                res = m.predict(source=im, imgsz=512, conf=min(a.det_conf), device=device, verbose=False)[0]
                r[tag] = [] if res.boxes is None or len(res.boxes) == 0 else [{"conf": float(y), "cls": names[int(k)]} for y, k in zip(res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int))]
            per[f"{g}/{p.name}"] = r
    md = ["# Flags outside the 25 classes: images with an accepted detection (= a wrongly assigned country)\n",
          "Two-stage: detector confidence td and classifier probability tc. Single-stage: confidence td, class other = rejection.\n",
          "| group | model | td | tc | accepted / n | rate | Wilson 95% |", "|---|---|---|---|---|---|---|"]
    summary = []
    for g in groups:
        rows = [v for v in per.values() if v["group"] == g]; n = len(rows)
        if not n: continue
        for td in a.det_conf:
            for tc in a.clf_prob:
                k = sum(any(d["conf"] >= td and d["prob"] >= tc for d in r["two_stage"]) for r in rows); lo, hi = wilson(k, n)
                md.append(f"| {g} | two-stage | {td} | {tc} | {k}/{n} | {k/n:.3f} | [{lo:.3f}, {hi:.3f}] |"); summary.append(dict(group=g, model="two_stage", td=td, tc=tc, k=k, n=n))
            for tag in singles:
                k = sum(any(d["conf"] >= td and d["cls"] != "other" for d in r[tag]) for r in rows); lo, hi = wilson(k, n)
                md.append(f"| {g} | {tag} | {td} | - | {k}/{n} | {k/n:.3f} | [{lo:.3f}, {hi:.3f}] |"); summary.append(dict(group=g, model=tag, td=td, tc=None, k=k, n=n))
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "per_image": per}, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))

if __name__ == "__main__":
    main()
