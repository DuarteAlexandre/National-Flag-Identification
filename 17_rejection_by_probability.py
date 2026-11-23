"""R2.1/R1.3: effect of rejecting detections whose classifier probability (max softmax of classifier B) is low.
For each pair (detector confidence, classifier probability) it reports end-to-end precision/recall on the maritime
test photographs (same matching as script 13) and the images with >=1 retained detection among the flag-free
images (the test negatives plus the extra negatives). Exploratory: thresholds are NOT tuned on separate data.
Run from the repository root:  python 17_rejection_by_probability.py --neg-dir <folder with extra negative images>"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from math import sqrt
from pathlib import Path
import numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("m13", ROOT / "13_run_maritime_experiments.py")
m13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m13)
from common import build_model, classify_pil_batch, crop_box, yolo_detect, load_rgb, pick_device, list_dataset
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass
from ultralytics import YOLO
EXT = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".avif", ".bmp"}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--det-conf", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--clf-prob", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--out", default="results/maritime/rejection_by_probability")
    a = ap.parse_args()
    device = pick_device(); _, classes = list_dataset()
    det = YOLO(str(ROOT / f"work/maritime_detector_seed{a.seed}/runs/train/weights/best.pt"))
    clf = build_model("resnet18", len(classes), pretrained=False)
    clf.load_state_dict(torch.load(ROOT / "results/maritime/classifier_B.pt", map_location="cpu")); clf = clf.to(device).eval()
    def run(im):
        b, c = yolo_detect(det, im, 512, min(a.det_conf), device)
        cr = [crop_box(im, x, a.margin) or im for x in b]
        pr = classify_pil_batch(clf, cr, device) if cr else np.zeros((0, len(classes)))
        return b, c, pr
    frames = m13.load_frames("test", len(classes))
    test = [(f,) + run(load_rgb(f["path"])) for f in frames]
    negs = [run(Image.open(p).convert("RGB"))[1:] for p in sorted(Path(a.neg_dir).iterdir()) if p.suffix.lower() in EXT]
    n_new = len(negs); n_old = sum(1 for f in frames if not f["boxes"])
    rows, md = [], [f"# Rejection by classifier probability (classifier B)\n",
                    f"Extra negatives: {n_new}; test negatives: {n_old}.\n",
                    "| detector conf | classifier prob >= | e2e precision | e2e recall | FA extra | FA test | FA pooled | Wilson 95% |", "|---|---|---|---|---|---|---|---|"]
    for td in a.det_conf:
        for tc in a.clf_prob:
            e2e_tp = n_scored = n_gt = fa_old = 0
            for f, b, c, pr in test:
                keep = [i for i in range(len(c)) if c[i] >= td and pr[i].max() >= tc]
                if not f["boxes"]:
                    fa_old += bool(keep)
                n_gt += sum(g["cls"] >= 0 for g in f["boxes"])
                if keep:
                    kb, kc = b[keep], c[keep]
                    pairs = m13.match(kb, kc, f["boxes"])
                    n_scored += len(keep) - sum(f["boxes"][g]["cls"] < 0 for g in pairs.values())
                    for d, g in pairs.items():
                        e2e_tp += f["boxes"][g]["cls"] >= 0 and int(pr[keep[d]].argmax()) == f["boxes"][g]["cls"]
            fa_new = sum(any(c[i] >= td and pr[i].max() >= tc for i in range(len(c))) for c, pr in negs)
            k, n = fa_new + fa_old, n_new + n_old; lo, hi = wilson(k, n)
            P = e2e_tp / n_scored if n_scored else float("nan"); R = e2e_tp / n_gt
            rows.append(dict(det_conf=td, clf_prob=tc, e2e_precision=P, e2e_recall=R, fa_extra=fa_new, fa_test=fa_old, fa_pooled=k, n_pooled=n, wilson95=[lo, hi]))
            md.append(f"| {td} | {tc} | {P:.3f} | {R:.3f} | {fa_new}/{n_new} | {fa_old}/{n_old} | {k}/{n} | [{lo:.3f}, {hi:.3f}] |")
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1), encoding="utf-8"); out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
