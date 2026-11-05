"""
13_run_maritime_experiments.py
Avaliação em CENAS MARÍTIMAS REAIS, sem recorte, com caixas anotadas manualmente e frames sem bandeira
(responde a R1.4, R1.8, R2.1, R3.1, R3.3, R3.5 e, no domínio marítimo, R1.6 / R3.4).

O que faz
  1. Treina os classificadores ResNet18 (augmentation 'targeted'):
        A = somente dataset interno (imagens recortadas)
        B = dataset interno + recortes REAIS (caixas anotadas) dos vídeos de TREINO marítimo
  2. Treina o detector YOLOv8n (classe única "flag") com as caixas REAIS dos vídeos de treino
     (substitui as caixas-proxy do artigo original). Reporta mAP@0.5 e mAP@0.5:0.95 (validação e TESTE).
  3. No TESTE (vídeos nunca vistos em nenhuma etapa): métricas de detecção, ponta-a-ponta (detector -> recorte ->
     classificador), "oráculo" (caixa anotada -> classificador), taxa de falso alarme nos frames SEM bandeira,
     baselines sem detector (frame inteiro / recorte central), varredura de limiar, subgrupos (tamanho e,
     se existir maritime/attributes.csv: borrado / ocluído / deformado) e tempo por frame.
  4. Salva figuras qualitativas (entrada + caixas anotadas/preditas) em results/maritime/qualitative/.

Uso:
    python 13_run_maritime_experiments.py
    python 13_run_maritime_experiments.py --det-epochs 30 --det-imgsz 512     # mais rápido em CPU
    python 13_run_maritime_experiments.py --skip-train-det                     # reaproveita o detector já treinado
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from common import (MARITIME_DIR, RESULTS_DIR, ROOT, WORK_DIR, ArrayDataset, build_cache, build_model,
                    center_crop_pil, classification_metrics, classify_pil_batch, crop_box, env_info, get_classes,
                    group_split, list_dataset, load_json, load_rgb, pick_device, save_json, seed_everything,
                    time_calls, to_cache_size, train_classifier, train_yolo, yolo_detect, yolo_val, make_transforms)

FRAMES, LABELS = MARITIME_DIR / "frames", MARITIME_DIR / "labels"
YOLO_DIR = MARITIME_DIR / "yolo_flag"
OUT = RESULTS_DIR / "maritime"
FALLBACK_WEAK = ["Germany", "Belgium", "Netherlands", "Canada", "Denmark"]


# ----------------------------------------------------------------------------
# dados marítimos
# ----------------------------------------------------------------------------
def load_frames(split, n_ds_classes):
    sov = {r["video_id"]: r["split"] for r in csv.DictReader(open(MARITIME_DIR / "split.csv", encoding="utf-8"))}
    out = []
    for lf in sorted(LABELS.glob("*.txt")):
        stem = lf.stem
        if sov.get(stem.rsplit("_", 1)[0]) != split or not (FRAMES / f"{stem}.jpg").exists():
            continue
        W, H = Image.open(FRAMES / f"{stem}.jpg").size
        boxes = []
        for line in lf.read_text().splitlines():
            if not line.strip():
                continue
            c, x, y, w, h = int(line.split()[0]), *map(float, line.split()[1:5])
            boxes.append({"cls": c if c < n_ds_classes else -1,           # -1 = "other"
                          "xyxy": [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                          "line": len(boxes), "bucket": None})
            s = (w * W * h * H) ** 0.5
            boxes[-1]["bucket"] = "small" if s < 32 else ("medium" if s < 96 else "large")
        out.append({"stem": stem, "path": FRAMES / f"{stem}.jpg", "W": W, "H": H, "boxes": boxes})
    return out


def gt_crops(frames, margin):
    imgs, ys = [], []
    for fr in frames:
        im = load_rgb(fr["path"])
        for b in fr["boxes"]:
            if b["cls"] >= 0:
                c = crop_box(im, b["xyxy"], margin)
                if c is not None:
                    imgs.append(to_cache_size(c))
                    ys.append(b["cls"])
    return (np.stack(imgs) if imgs else np.zeros((0, 256, 256, 3), np.uint8)), np.array(ys, dtype=int)


def final_weak(classes, k):
    cnt = Counter()
    for f in (RESULTS_DIR / "internal").glob("seed*/weak_classes.json"):
        cnt.update(load_json(f)["weak_classes"])
    if not cnt:
        print("AVISO: sem results/internal/seed*/weak_classes.json; usando as 5 classes originais como 'fracas'.")
        return [classes.index(c) for c in FALLBACK_WEAK]
    top = sorted(cnt, key=lambda c: (-cnt[c], c))[:k]
    return [classes.index(c) for c in top]


# ----------------------------------------------------------------------------
# classificadores
# ----------------------------------------------------------------------------
def get_classifier(tag, classes, args, device, int_imgs, int_lbls, groups, mar_train, mar_val):
    f = OUT / f"classifier_{tag}.pt"
    model = build_model("resnet18", len(classes), pretrained=not args.no_pretrained)
    if f.exists() and not args.force_retrain:
        model.load_state_dict(torch.load(f, map_location="cpu"))
        return model.to(device).eval()
    seed_everything(args.seed)
    tr, va, te = group_split(int_lbls, groups, seed=0)
    tr = np.concatenate([tr, te])        # o "teste interno" não é usado aqui: o teste real é o marítimo
    Xtr, ytr, Xva, yva = int_imgs[tr], int_lbls[tr], int_imgs[va], int_lbls[va]
    if tag == "B":
        Xtr, ytr = np.concatenate([Xtr, mar_train[0]]), np.concatenate([ytr, mar_train[1]])
        Xva, yva = np.concatenate([Xva, mar_val[0]]), np.concatenate([yva, mar_val[1]])
    weak = final_weak(classes, args.weak_k)
    print(f"\n[classificador {tag}] treino {len(ytr)} | val {len(yva)} | classes fracas: {[classes[w] for w in weak]}")
    tds = ArrayDataset(Xtr, ytr, aug="targeted", weak=weak, repeat=args.repeat, train=True, img_size=args.img_size)
    vds = ArrayDataset(Xva, yva, img_size=args.img_size)
    model, hist = train_classifier(model, tds, vds, device, epochs=args.epochs, patience=args.patience,
                                   lr=args.lr, batch=args.batch)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f)
    save_json({k: v for k, v in hist.items()} | {"weak_classes": [classes[w] for w in weak]}, OUT / f"classifier_{tag}_training.json")
    return model.eval()


# ----------------------------------------------------------------------------
# casamento detecção x anotação
# ----------------------------------------------------------------------------
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(det_boxes, det_confs, gts, thr=0.5):
    """Guloso por confiança decrescente. Devolve {idx_det: idx_gt}."""
    pairs, used = {}, set()
    for d in np.argsort(-np.asarray(det_confs)):
        best, bi = thr, -1
        for g, gb in enumerate(gts):
            if g in used:
                continue
            v = iou(det_boxes[d], gb["xyxy"])
            if v >= best:
                best, bi = v, g
        if bi >= 0:
            pairs[int(d)] = bi
            used.add(bi)
    return pairs


def evaluate(frames, cache, tag, conf, classes):
    """cache[stem] = dict(boxes, confs, pred_cls[tag], oracle[tag])"""
    tp = fp = fn = 0
    cls_ok = cls_n = e2e_tp = n_dets_scored = n_gt_known = 0
    neg_frames = neg_alarm = neg_fp = 0
    gt_rows, frame_flags = [], {}
    for fr in frames:
        c = cache[fr["stem"]]
        sel = np.where(c["confs"] >= conf)[0]
        boxes, confs, preds = c["boxes"][sel], c["confs"][sel], c["pred_cls"][tag][sel]
        pairs = match(boxes, confs, fr["boxes"])
        matched_g = set(pairs.values())
        f_fp, f_fn, f_wrong = len(sel) - len(pairs), len(fr["boxes"]) - len(pairs), 0
        tp += len(pairs)
        fp += f_fp
        fn += f_fn
        n_gt_known += sum(b["cls"] >= 0 for b in fr["boxes"])
        n_dets_scored += len(sel) - sum(fr["boxes"][g]["cls"] < 0 for g in pairs.values())
        for d, g in pairs.items():
            gt = fr["boxes"][g]
            if gt["cls"] >= 0:
                cls_n += 1
                ok = int(preds[d]) == gt["cls"]
                cls_ok += ok
                e2e_tp += ok
                f_wrong += (not ok)
        for g, gt in enumerate(fr["boxes"]):
            det = g in matched_g
            d_idx = next((d for d, gg in pairs.items() if gg == g), None)
            gt_rows.append({"image": fr["stem"], "line": gt["line"], "country": classes[gt["cls"]] if gt["cls"] >= 0 else "other",
                            "bucket": gt["bucket"], "detected": det,
                            "e2e_correct": bool(det and gt["cls"] >= 0 and int(preds[d_idx]) == gt["cls"]),
                            "oracle_correct": (int(c["oracle"][tag][g]) == gt["cls"]) if gt["cls"] >= 0 else None})
        if not fr["boxes"]:
            neg_frames += 1
            neg_alarm += int(len(sel) > 0)
            neg_fp += len(sel)
        frame_flags[fr["stem"]] = {"fp": f_fp, "fn": f_fn, "wrong": f_wrong, "n_det": len(sel)}
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "conf": conf, "det_tp": tp, "det_fp": fp, "det_fn": fn, "det_precision": P, "det_recall": R,
        "det_f1": 2 * P * R / (P + R) if P + R > 0 else float("nan"),
        "class_acc_on_matched": cls_ok / cls_n if cls_n else float("nan"),
        "e2e_precision": e2e_tp / n_dets_scored if n_dets_scored else float("nan"),
        "e2e_recall": e2e_tp / n_gt_known if n_gt_known else float("nan"),
        "neg_frames": neg_frames, "neg_frames_with_false_alarm": neg_alarm,
        "false_alarm_rate_neg_frames": neg_alarm / neg_frames if neg_frames else float("nan"),
        "fp_per_neg_frame": neg_fp / neg_frames if neg_frames else float("nan"),
    }, gt_rows, frame_flags


def subgroup_table(gt_rows, attrs):
    df = pd.DataFrame(gt_rows)
    if df.empty:
        return pd.DataFrame()
    if attrs is not None:
        df = df.merge(attrs, on=["image", "line"], how="left")
    groups = {f"size={b}": df["bucket"] == b for b in ("small", "medium", "large")}
    if attrs is not None:
        for a in ("blurred", "occluded", "deformed"):
            groups[f"{a}=1"] = df[a] == 1
        groups["clean (none of blurred/occluded/deformed)"] = (df["blurred"] == 0) & (df["occluded"] == 0) & (df["deformed"] == 0)
    rows = []
    for name, m in groups.items():
        d = df[m]
        known = d[d["country"] != "other"]
        rows.append({"subgroup": name, "n_boxes": len(d), "detection_recall": d["detected"].mean() if len(d) else np.nan,
                     "e2e_recall": known["e2e_correct"].mean() if len(known) else np.nan,
                     "oracle_accuracy": known["oracle_correct"].astype(float).mean() if len(known) else np.nan})
    return pd.DataFrame(rows)


def draw(fr, boxes, confs, preds, names, path, width=1100):
    im = load_rgb(fr["path"])
    s = width / im.width
    im = im.resize((width, int(im.height * s)), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    for b in fr["boxes"]:
        x1, y1, x2, y2 = [v * s for v in b["xyxy"]]
        d.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
        d.text((x1 + 2, max(0, y1 - 12)), "GT " + (names[b["cls"]] if b["cls"] >= 0 else "other"), fill=(0, 255, 0))
    for bx, cf, pc in zip(boxes, confs, preds):
        x1, y1, x2, y2 = [v * s for v in bx]
        d.rectangle([x1, y1, x2, y2], outline=(230, 30, 30), width=2)
        d.text((x1 + 2, min(im.height - 12, y2 + 2)), f"{names[int(pc)]} {cf:.2f}", fill=(255, 60, 60))
    im.save(path, quality=90)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--det-epochs", type=int, default=40)
    ap.add_argument("--det-imgsz", type=int, default=640)
    ap.add_argument("--det-batch", type=int, default=8)
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="limiar operacional")
    ap.add_argument("--conf-sweep", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.7])
    ap.add_argument("--margin", type=float, default=0.10, help="margem relativa nos recortes")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--weak-k", type=int, default=5)
    ap.add_argument("--skip-train-det", action="store_true")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="(apenas teste de fumaça)")
    ap.add_argument("--n-viz", type=int, default=8)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    items, classes = list_dataset()
    paths = [p for p, _ in items]
    int_lbls = np.array([y for _, y in items])
    names_m = [l.strip() for l in (MARITIME_DIR / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert names_m[:len(classes)] == classes, "maritime/classes.txt não coincide com as pastas de dataset/"
    groups = pd.read_csv(ROOT / "splits" / "groups.csv").set_index("path").loc[paths, "group_id"].values
    int_imgs = build_cache(paths, lambda im: im, "direct_256")

    fr_train, fr_val, fr_test = (load_frames(s, len(classes)) for s in ("train", "val", "test"))
    print(f"frames marítimos: treino {len(fr_train)} | val {len(fr_val)} | teste {len(fr_test)}")
    if not fr_test or not fr_train:
        raise SystemExit("Faltam frames de treino/teste. Rode 12_prepare_maritime.py.")

    # ---------------- 1) classificadores
    mar_tr, mar_va = gt_crops(fr_train, args.margin), gt_crops(fr_val, args.margin)
    print(f"recortes anotados (países conhecidos): treino {len(mar_tr[1])} | val {len(mar_va[1])}")
    clfs = {t: get_classifier(t, classes, args, device, int_imgs, int_lbls, groups, mar_tr, mar_va) for t in ("A", "B")}

    # ---------------- 2) detector
    from ultralytics import YOLO
    det_root = WORK_DIR / f"maritime_detector_seed{args.seed}"
    best = det_root / "runs" / "train" / "weights" / "best.pt"
    if not (args.skip_train_det and best.exists()):
        print(f"\n[detector] treinando YOLOv8n ({args.det_epochs} épocas, imgsz {args.det_imgsz}) com caixas REAIS ...")
        best = train_yolo(YOLO_DIR / "dataset.yaml", det_root / "runs", "train", args.yolo_weights, args.det_epochs,
                          args.det_imgsz, args.det_batch, args.seed, device, workers=0, patience=15)
    det_val = yolo_val(best, YOLO_DIR / "dataset.yaml", "val", args.det_imgsz, device, det_root / "runs", "val")
    det_test = yolo_val(best, YOLO_DIR / "dataset.yaml", "test", args.det_imgsz, device, det_root / "runs", "test")
    print("detector  val:", det_val, "\ndetector teste:", det_test)
    ymodel = YOLO(str(best))

    # ---------------- 3) inferência única no teste (conf mínimo da varredura)
    cmin = min(args.conf_sweep + [args.conf])
    cache, pil = {}, {}
    for fr in fr_test:
        im = load_rgb(fr["path"])
        pil[fr["stem"]] = im
        boxes, confs = yolo_detect(ymodel, im, args.det_imgsz, cmin, device)
        crops = [crop_box(im, b, args.margin) or im for b in boxes]
        entry = {"boxes": boxes, "confs": confs, "pred_cls": {}, "oracle": {}, "pred_prob": {}}
        gt_c = [crop_box(im, b["xyxy"], args.margin) or im for b in fr["boxes"]]
        for t, m in clfs.items():
            pr = classify_pil_batch(m, crops, device, args.img_size) if crops else np.zeros((0, len(classes)))
            entry["pred_cls"][t] = pr.argmax(1) if len(pr) else np.zeros(0, dtype=int)
            entry["pred_prob"][t] = pr.max(1) if len(pr) else np.zeros(0)
            po = classify_pil_batch(m, gt_c, device, args.img_size) if gt_c else np.zeros((0, len(classes)))
            entry["oracle"][t] = po.argmax(1) if len(po) else np.zeros(0, dtype=int)
        cache[fr["stem"]] = entry

    attrs = None
    if (MARITIME_DIR / "attributes.csv").exists():
        attrs = pd.read_csv(MARITIME_DIR / "attributes.csv").rename(columns={"image": "image"})[["image", "line", "blurred", "occluded", "deformed"]]
        attrs = attrs.fillna(0).astype({"blurred": int, "occluded": int, "deformed": int})
    else:
        print("(maritime/attributes.csv não encontrado: subgrupos blurred/occluded/deformed serão omitidos)")

    # ---------------- 4) métricas
    results = {"detector_val": det_val, "detector_test": det_test, "n_test_frames": len(fr_test),
               "n_test_boxes": sum(len(f["boxes"]) for f in fr_test),
               "n_test_negative_frames": sum(1 for f in fr_test if not f["boxes"]),
               "operating_conf": args.conf, "env": env_info(), "args": vars(args), "classifiers": {}}
    md = []
    tf = make_transforms(args.img_size)["eval"]
    for t, m in clfs.items():
        res = {}
        # oráculo
        yt, yp = [], []
        for fr in fr_test:
            for g, b in enumerate(fr["boxes"]):
                if b["cls"] >= 0:
                    yt.append(b["cls"])
                    yp.append(int(cache[fr["stem"]]["oracle"][t][g]))
        res["oracle"] = classification_metrics(yt, yp, classes, present_only=True) if yt else None
        # ponta-a-ponta e varredura
        res["sweep"] = {}
        for c in sorted(set(args.conf_sweep + [args.conf])):
            r, gt_rows, flags = evaluate(fr_test, cache, t, c, classes)
            res["sweep"][str(c)] = r
            if c == args.conf:
                res["operating"], op_rows, op_flags = r, gt_rows, flags
        res["subgroups"] = subgroup_table(op_rows, attrs).to_dict("records")
        # baselines SEM detector (nível de frame): top-1 pertence ao conjunto de países anotados no frame?
        base = {"direct_full_frame": [], "center_crop_80": [], "yolo_top1_box": []}
        for fr in fr_test:
            gts = {b["cls"] for b in fr["boxes"] if b["cls"] >= 0}
            if not gts:
                continue
            im = pil[fr["stem"]]
            base["direct_full_frame"].append(int(classify_pil_batch(m, [im], device, args.img_size)[0].argmax()) in gts)
            base["center_crop_80"].append(int(classify_pil_batch(m, [center_crop_pil(im, 0.8)], device, args.img_size)[0].argmax()) in gts)
            e = cache[fr["stem"]]
            sel = np.where(e["confs"] >= args.conf)[0]
            base["yolo_top1_box"].append(bool(len(sel)) and int(e["pred_cls"][t][sel[int(np.argmax(e["confs"][sel]))]]) in gts)
        res["frame_level_top1_accuracy"] = {k: float(np.mean(v)) if v else float("nan") for k, v in base.items()}
        res["frame_level_n"] = len(base["direct_full_frame"])
        results["classifiers"][t] = res

        # ---- tempo por frame (lote=1, frame já decodificado)
        ims = [pil[f["stem"]] for f in fr_test]

        def full(im, m=m):
            b, cf = yolo_detect(ymodel, im, args.det_imgsz, args.conf, device)
            cr = [crop_box(im, x, args.margin) or im for x in b]
            return classify_pil_batch(m, cr, device, args.img_size) if cr else None
        if t == "B":
            results["timing_per_frame"] = {
                "detector_only": time_calls(lambda im: yolo_detect(ymodel, im, args.det_imgsz, args.conf, device), ims),
                "detector_plus_classifier": time_calls(full, ims),
                "protocol": f"dispositivo={device}, lote=1 frame, frame já decodificado (exclui leitura de disco), imgsz={args.det_imgsz}, "
                            f"conf={args.conf}; inclui NMS, recortes, pré-processamento e classificação de todas as caixas detectadas"}

        # ---- figuras qualitativas (usa o classificador B)
        if t == "B":
            qdir = OUT / "qualitative"
            qdir.mkdir(exist_ok=True)
            bad = [f for f in fr_test if op_flags[f["stem"]]["fp"] or op_flags[f["stem"]]["fn"] or op_flags[f["stem"]]["wrong"]]
            ok = [f for f in fr_test if f["boxes"] and f not in bad]
            rng = np.random.default_rng(0)
            for tag_q, pool in (("ok", ok), ("fail", bad)):
                for k, i in enumerate(rng.permutation(len(pool))[:args.n_viz]):
                    fr = pool[i]
                    e = cache[fr["stem"]]
                    sel = np.where(e["confs"] >= args.conf)[0]
                    draw(fr, e["boxes"][sel], e["confs"][sel], e["pred_cls"][t][sel], classes, qdir / f"{tag_q}_{k + 1:02d}_{fr['stem']}.jpg")

    # ---------------- 5) tabelas em Markdown
    md += [f"Teste marítimo: {results['n_test_frames']} frames, {results['n_test_boxes']} caixas anotadas, "
           f"{results['n_test_negative_frames']} frames sem bandeira.", "",
           "**Detector (caixas reais, classe única)**", "",
           "| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |", "|---|---|---|---|---|"]
    for k, d in (("val", det_val), ("test", det_test)):
        md.append(f"| {k} | {d['map50']:.3f} | {d['map50_95']:.3f} | {d['precision']:.3f} | {d['recall']:.3f} |")
    for t in clfs:
        r = results["classifiers"][t]
        o = r["operating"]
        md += ["", f"**Classificador {t}** (limiar operacional conf={args.conf})", "",
               "| Métrica | Valor |", "|---|---|",
               f"| Detecção: precision / recall / F1 (IoU≥0.5) | {o['det_precision']:.3f} / {o['det_recall']:.3f} / {o['det_f1']:.3f} |",
               f"| Acurácia de país nas caixas casadas | {o['class_acc_on_matched']:.3f} |",
               f"| Ponta-a-ponta: precision / recall | {o['e2e_precision']:.3f} / {o['e2e_recall']:.3f} |",
               f"| Falso alarme em frames sem bandeira (fração) / FP por frame negativo | {o['false_alarm_rate_neg_frames']:.3f} / {o['fp_per_neg_frame']:.3f} |"]
        if r["oracle"]:
            md.append(f"| Oráculo (caixa anotada→classificador): acurácia / F1 macro (n={r['oracle']['n']}) | "
                      f"{r['oracle']['accuracy']:.3f} / {r['oracle']['f1_macro']:.3f} |")
        fl = r["frame_level_top1_accuracy"]
        md += ["", f"Acurácia top-1 por frame (n={r['frame_level_n']} frames com bandeira) — frame inteiro: {fl['direct_full_frame']:.3f} | "
               f"recorte central 80%: {fl['center_crop_80']:.3f} | caixa do YOLO de maior confiança: {fl['yolo_top1_box']:.3f}", "",
               "Varredura de limiar:", "", "| conf | det P | det R | e2e P | e2e R | falso alarme (frames neg.) |", "|---|---|---|---|---|---|"]
        for c, s in r["sweep"].items():
            md.append(f"| {c} | {s['det_precision']:.3f} | {s['det_recall']:.3f} | {s['e2e_precision']:.3f} | {s['e2e_recall']:.3f} | {s['false_alarm_rate_neg_frames']:.3f} |")
        if r["subgroups"]:
            md += ["", "Subgrupos:", "", "| Subgrupo | n caixas | det. recall | e2e recall | acurácia oráculo |", "|---|---|---|---|---|"]
            for s in r["subgroups"]:
                md.append(f"| {s['subgroup']} | {s['n_boxes']} | {s['detection_recall']:.3f} | {s['e2e_recall']:.3f} | {s['oracle_accuracy']:.3f} |")
    tp_ = results["timing_per_frame"]
    md += ["", f"**Tempo por frame** ({tp_['protocol']}): detector {tp_['detector_only']['mean_ms']:.1f} ± {tp_['detector_only']['std_ms']:.1f} ms; "
           f"detector+classificador {tp_['detector_plus_classifier']['mean_ms']:.1f} ± {tp_['detector_plus_classifier']['std_ms']:.1f} ms"]
    save_json(results, OUT / "maritime_results.json")
    (OUT / "maritime_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nResultados em {OUT}  (figuras em {OUT / 'qualitative'})")


if __name__ == "__main__":
    main()
