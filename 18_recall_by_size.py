"""R1.1: detection and end-to-end recall of the maritime test boxes by flag size.
Size = sqrt(box area) in pixels of the 512-px-long-side image seen by the detector (letterbox scale 512/max(W,H)),
and also in original pixels. Part B simulates distance: each test photograph is shrunk by a factor s and pasted on a
grey canvas of the original size (the flag becomes s times smaller); recall is reported by effective size.
Run from the repository root:  python 18_recall_by_size.py"""
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
from ultralytics import YOLO

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def bucket(s): return "<16" if s < 16 else "16-32" if s < 32 else "32-64" if s < 64 else "64-96" if s < 96 else ">=96"
ORDER = ["<16", "16-32", "32-64", "64-96", ">=96"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.35, 0.25, 0.18])
    ap.add_argument("--out", default="results/maritime/recall_by_size")
    a = ap.parse_args()
    device = pick_device(); _, classes = list_dataset()
    det = YOLO(str(ROOT / f"work/maritime_detector_seed{a.seed}/runs/train/weights/best.pt"))
    clf = build_model("resnet18", len(classes), pretrained=False)
    clf.load_state_dict(torch.load(ROOT / "results/maritime/classifier_B.pt", map_location="cpu")); clf = clf.to(device).eval()
    res = {"sizes": {}, "by_scale": {}}
    for split in ("train", "val", "test"):
        fr = m13.load_frames(split, len(classes))
        s512 = [sqrt((b["xyxy"][2]-b["xyxy"][0])*(b["xyxy"][3]-b["xyxy"][1])) * 512 / max(f["W"], f["H"]) for f in fr for b in f["boxes"]]
        res["sizes"][split] = {k: sum(bucket(s) == k for s in s512) for k in ORDER}; res["sizes"][split]["n"] = len(s512)
        res["sizes"][split]["min"] = float(min(s512)); res["sizes"][split]["median"] = float(np.median(s512))
    test = m13.load_frames("test", len(classes))
    for s in a.scales:
        cnt = {k: [0, 0, 0] for k in ORDER}   # n, detected, e2e correct
        for f in test:
            if not f["boxes"]: continue
            im = load_rgb(f["path"]); W, H = im.size
            if s < 1.0:
                small = im.resize((max(1, int(W * s)), max(1, int(H * s))), Image.BILINEAR)
                canvas = Image.new("RGB", (W, H), (114, 114, 114)); ox, oy = (W - small.size[0]) // 2, (H - small.size[1]) // 2
                canvas.paste(small, (ox, oy)); im2 = canvas
                boxes_gt = [dict(b, xyxy=[b["xyxy"][0]*s+ox, b["xyxy"][1]*s+oy, b["xyxy"][2]*s+ox, b["xyxy"][3]*s+oy]) for b in f["boxes"]]
            else:
                im2, boxes_gt = im, f["boxes"]
            b, c = yolo_detect(det, im2, 512, a.conf, device)
            cr = [crop_box(im2, x, a.margin) or im2 for x in b]
            pr = classify_pil_batch(clf, cr, device) if cr else np.zeros((0, len(classes)))
            pairs = m13.match(b, c, boxes_gt)
            inv = {g: d for d, g in pairs.items()}
            for g, gt in enumerate(boxes_gt):
                if gt["cls"] < 0: continue
                size = sqrt((gt["xyxy"][2]-gt["xyxy"][0])*(gt["xyxy"][3]-gt["xyxy"][1])) * 512 / max(W, H)
                k = bucket(size); cnt[k][0] += 1
                if g in inv:
                    cnt[k][1] += 1; cnt[k][2] += int(pr[inv[g]].argmax()) == gt["cls"]
        res["by_scale"][str(s)] = cnt
    # pooled over all scales, by effective size
    pooled = {k: [0, 0, 0] for k in ORDER}
    for s, cnt in res["by_scale"].items():
        for k in ORDER:
            for i in range(3): pooled[k][i] += cnt[k][i]
    res["pooled_by_effective_size"] = pooled
    md = ["# Recall by flag size (sqrt of box area in pixels, 512-px long side)\n", "## A. Test boxes as photographed (scale 1.0)\n",
          "| size (px) | boxes | detected | detection recall (95% CI) | end-to-end recall |", "|---|---|---|---|---|"]
    def line(k, v):
        n, d, e = v; lo, hi = wilson(d, n)
        return f"| {k} | {n} | {d} | {d/n:.3f} ({lo:.2f}-{hi:.2f}) | {e/n:.3f} |" if n else f"| {k} | 0 | - | - | - |"
    for k in ORDER: md.append(line(k, res["by_scale"]["1.0"][k]))
    md += ["\n## B. Simulated distance: all scales pooled by effective size\n", "| effective size (px) | boxes | detected | detection recall (95% CI) | end-to-end recall |", "|---|---|---|---|---|"]
    for k in ORDER: md.append(line(k, pooled[k]))
    md.append("\n## Size distribution of annotated boxes (512-px scale)\n")
    for sp, v in res["sizes"].items(): md.append(f"- {sp}: n={v['n']}, min={v['min']:.1f}, median={v['median']:.1f}, " + ", ".join(f"{k}: {v[k]}" for k in ORDER))
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8"); out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()
