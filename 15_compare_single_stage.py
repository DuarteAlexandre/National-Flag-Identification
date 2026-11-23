"""
15_compare_single_stage.py
Comparação do pipeline de DOIS ESTÁGIOS (detector de bandeira + ResNet18) com um YOLOv8n de ESTÁGIO ÚNICO treinado nas
mesmas imagens marítimas para prever localização E país (resposta a R1.2, 2a rodada).

O que faz
  1. Monta maritime/yolo_multi/ (cópia das fotos marítimas de treino/validação/teste e dos rótulos com a classe de país,
     26 classes = 25 países + "other"; os mesmos splits e as mesmas caixas anotadas usadas nos demais experimentos).
  2. Treina YOLOv8n multiclasse a partir de yolov8n.pt (mesmas imagens, imgsz 512, lote 8, semente 0) com cada número de épocas
     de --epochs-list (padrão: 30, igual ao detector de classe única, e 100, para dar ao estágio único mais tempo de
     treinamento). Reporta mAP@0.5 e mAP@0.5:0.95 (multiclasse) no teste.
  3. No TESTE marítimo, para o estágio único e para o pipeline de dois estágios (detector de classe única + classificador B,
     ResNet18, pesos já existentes): detecção (IoU >= 0.5), desempenho ponta a ponta (localização E país corretos),
     acurácia top-1 por frame, falsos alarmes nos quadros sem bandeira e varredura de limiar. Diferenças pareadas por caixa
     (IC bootstrap 95%) entre cada estágio único e o de dois estágios.
  4. Mede o tempo por frame (lote 1, frame já decodificado, CPU) dos dois esquemas, em passagens alternadas.

Uso (na pasta do projeto, onde estão common.py e 13_run_maritime_experiments.py):
    python 15_compare_single_stage.py
    python 15_compare_single_stage.py --epochs-list 1 --yolo-weights yolov8n.pt     # teste de fumaça
Saídas: results/maritime/single_stage_comparison.json e .md
"""
import argparse
import importlib.util
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (MARITIME_DIR, RESULTS_DIR, WORK_DIR, build_model, classify_pil_batch, crop_box, env_info, list_dataset,
                    load_rgb, pick_device, save_json, train_yolo, write_yolo_yaml, yolo_detect, yolo_val)

_spec = importlib.util.spec_from_file_location("m13", Path(__file__).resolve().parent / "13_run_maritime_experiments.py")
m13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m13)

OUT = RESULTS_DIR / "maritime"
YOLO_MULTI = MARITIME_DIR / "yolo_multi"


def build_multi_dataset(names, frames):
    if YOLO_MULTI.exists():
        shutil.rmtree(YOLO_MULTI)
    for sp, frs in frames.items():
        (YOLO_MULTI / "images" / sp).mkdir(parents=True, exist_ok=True)
        (YOLO_MULTI / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for fr in frs:
            shutil.copy2(fr["path"], YOLO_MULTI / "images" / sp / f"{fr['stem']}.jpg")
            lf = m13.LABELS / f"{fr['stem']}.txt"
            dst = YOLO_MULTI / "labels" / sp / f"{fr['stem']}.txt"
            if lf.exists():
                shutil.copy2(lf, dst)
            else:
                dst.write_text("")
    write_yolo_yaml(YOLO_MULTI / "dataset.yaml", YOLO_MULTI, names=tuple(names), test="images/test")


def detect_cls(model, im, imgsz, conf, device):
    r = model.predict(source=im, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)


def boot_diff(a, b, n=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs-list", type=int, nargs="+", default=[30, 100])
    ap.add_argument("--det-imgsz", type=int, default=512)
    ap.add_argument("--det-batch", type=int, default=8)
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-sweep", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.7])
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--timing-passes", type=int, default=3)
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    _, classes = list_dataset()
    names = [l.strip() for l in (MARITIME_DIR / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert names[:len(classes)] == classes
    fr_train, fr_val, fr_test = (m13.load_frames(s, len(classes)) for s in ("train", "val", "test"))
    print(f"frames: treino {len(fr_train)} | val {len(fr_val)} | teste {len(fr_test)}; classes do YOLO multiclasse: {len(names)}")
    build_multi_dataset(names, {"train": fr_train, "val": fr_val, "test": fr_test})

    from ultralytics import YOLO
    # ---------------- detector de classe único + classificador B (dois estágios; pesos já existentes)
    det_best = WORK_DIR / f"maritime_detector_seed{args.seed}" / "runs" / "train" / "weights" / "best.pt"
    det = YOLO(str(det_best))
    clf = build_model("resnet18", len(classes), pretrained=False)
    clf.load_state_dict(torch.load(OUT / "classifier_B.pt", map_location="cpu"))
    clf = clf.to(device).eval()

    # ---------------- YOLO multiclasse (estágio único)
    singles, map_res = {}, {}
    for ep in args.epochs_list:
        root = WORK_DIR / f"maritime_single_stage_seed{args.seed}_e{ep}"
        best = root / "runs" / "train" / "weights" / "best.pt"
        if args.force_retrain or not best.exists():
            print(f"\n[estágio único] treinando YOLOv8n multiclasse, {ep} épocas, imgsz {args.det_imgsz} ...")
            best = train_yolo(YOLO_MULTI / "dataset.yaml", root / "runs", "train", args.yolo_weights, ep, args.det_imgsz,
                              args.det_batch, args.seed, device, workers=0, patience=15 if ep <= 30 else 30)
        tag = f"single_e{ep}"
        singles[tag] = YOLO(str(best))
        map_res[tag] = {"val": yolo_val(best, YOLO_MULTI / "dataset.yaml", "val", args.det_imgsz, device, root / "runs", "val"),
                        "test": yolo_val(best, YOLO_MULTI / "dataset.yaml", "test", args.det_imgsz, device, root / "runs", "test")}
        print(tag, map_res[tag])

    # ---------------- inferência única no teste (limiar mínimo da varredura)
    cmin = min(args.conf_sweep + [args.conf])
    cache, pil = {}, {}
    for fr in fr_test:
        im = load_rgb(fr["path"])
        pil[fr["stem"]] = im
        ngt = len(fr["boxes"])
        e = {"pred_cls": {}, "oracle": {}, "boxes_by": {}, "confs_by": {}}
        b, c = yolo_detect(det, im, args.det_imgsz, cmin, device)
        crops = [crop_box(im, x, args.margin) or im for x in b]
        pr = classify_pil_batch(clf, crops, device, args.img_size) if crops else np.zeros((0, len(classes)))
        e["boxes_by"]["two_stage"], e["confs_by"]["two_stage"] = b, c
        e["pred_cls"]["two_stage"] = pr.argmax(1) if len(pr) else np.zeros(0, dtype=int)
        for tag, sm in singles.items():
            b, c, k = detect_cls(sm, im, args.det_imgsz, cmin, device)
            e["boxes_by"][tag], e["confs_by"][tag], e["pred_cls"][tag] = b, c, k
        for t in e["pred_cls"]:
            e["oracle"][t] = -np.ones(ngt, dtype=int)
        cache[fr["stem"]] = e

    results = {"env": env_info(), "args": vars(args), "n_test_frames": len(fr_test),
               "n_test_boxes": sum(len(f["boxes"]) for f in fr_test), "map_multiclass": map_res, "per_model": {}}
    per_box = {}
    for tag in ["two_stage"] + list(singles):
        view = {s: {"boxes": c["boxes_by"][tag], "confs": c["confs_by"][tag], "pred_cls": {tag: c["pred_cls"][tag]},
                    "oracle": {tag: c["oracle"][tag]}} for s, c in cache.items()}
        res = {"sweep": {}}
        for cf in sorted(set(args.conf_sweep + [args.conf])):
            r, rows, _ = m13.evaluate(fr_test, view, tag, cf, classes)
            res["sweep"][str(cf)] = r
            if cf == args.conf:
                res["operating"] = r
                df = pd.DataFrame(rows)
                per_box[tag] = df[df["country"] != "other"]["e2e_correct"].astype(int).values
        top1 = []
        for fr in fr_test:
            gts = {b["cls"] for b in fr["boxes"] if b["cls"] >= 0}
            if not gts:
                continue
            e = view[fr["stem"]]
            sel = np.where(e["confs"] >= args.conf)[0]
            top1.append(bool(len(sel)) and int(e["pred_cls"][tag][sel[int(np.argmax(e["confs"][sel]))]]) in gts)
        res["frame_top1"] = float(np.mean(top1))
        results["per_model"][tag] = res
    results["paired_vs_two_stage"] = {}
    for tag in singles:
        d, lo, hi = boot_diff(per_box["two_stage"], per_box[tag])
        results["paired_vs_two_stage"][tag] = {
            "e2e_recall_diff": d, "ci95": [lo, hi],
            "only_single_correct": int(((per_box["two_stage"] == 0) & (per_box[tag] == 1)).sum()),
            "only_two_stage_correct": int(((per_box["two_stage"] == 1) & (per_box[tag] == 0)).sum()), "n": int(len(per_box[tag]))}

    # ---------------- tempo por frame (passagens alternadas)
    ims = [pil[f["stem"]] for f in fr_test]
    runners = {"two_stage": lambda im: _two_stage(det, clf, im, args, device)}
    for tag, sm in singles.items():
        runners[tag] = (lambda im, sm=sm: detect_cls(sm, im, args.det_imgsz, args.conf, device))
    for fn in runners.values():
        for im in ims[:3]:
            fn(im)
    ts = {k: [] for k in runners}
    for _ in range(args.timing_passes):
        for k, fn in runners.items():
            for im in ims:
                t0 = time.perf_counter()
                fn(im)
                ts[k].append((time.perf_counter() - t0) * 1000)
    results["timing_ms_per_frame"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "median": float(np.median(v)),
                                          "n": len(v)} for k, v in ts.items()}
    results["timing_protocol"] = (f"dispositivo={device}, lote=1, frame ja decodificado, imgsz={args.det_imgsz}, conf={args.conf}, "
                                  f"{args.timing_passes} passagens alternadas; dois estagios = detector + NMS + recortes + "
                                  f"classificacao de todas as caixas; estagio unico = uma passagem do YOLO multiclasse com NMS")
    save_json(results, OUT / "single_stage_comparison.json")

    md = [f"Teste marítimo: {results['n_test_frames']} frames, {results['n_test_boxes']} caixas; limiar {args.conf}.", "",
          "| Esquema | Det. P / R | E2E P / R | Top-1 por frame | Falso alarme (quadros sem bandeira) | mAP@0.5 (multi) | mAP@0.5:0.95 (multi) | ms/frame |",
          "|---|---|---|---|---|---|---|---|"]
    for tag, res in results["per_model"].items():
        o = res["operating"]
        mp = map_res.get(tag, {}).get("test")
        t = results["timing_ms_per_frame"][tag]
        md.append(f"| {tag} | {o['det_precision']:.3f} / {o['det_recall']:.3f} | {o['e2e_precision']:.3f} / {o['e2e_recall']:.3f} | "
                  f"{res['frame_top1']:.3f} | {o['false_alarm_rate_neg_frames']:.2f} | "
                  f"{(mp['map50'] if mp else float('nan')):.3f} | {(mp['map50_95'] if mp else float('nan')):.3f} | "
                  f"{t['mean']:.1f} ± {t['sd']:.1f} (med {t['median']:.1f}) |")
    md += ["", "Diferença pareada do recall ponta a ponta (estágio único − dois estágios), IC bootstrap 95%:", "",
           "| Estágio único | Diferença | IC 95% | só estágio único acerta | só dois estágios acerta | n |", "|---|---|---|---|---|---|"]
    for tag, v in results["paired_vs_two_stage"].items():
        md.append(f"| {tag} | {v['e2e_recall_diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                  f"{v['only_single_correct']} | {v['only_two_stage_correct']} | {v['n']} |")
    md += ["", "Varredura de limiar (E2E P / R):", "", "| Esquema | " + " | ".join(str(c) for c in sorted(set(args.conf_sweep + [args.conf]))) + " |",
           "|---|" + "---|" * len(set(args.conf_sweep + [args.conf]))]
    for tag, res in results["per_model"].items():
        md.append(f"| {tag} | " + " | ".join(f"{s['e2e_precision']:.3f} / {s['e2e_recall']:.3f}" for s in res["sweep"].values()) + " |")
    md += ["", results["timing_protocol"]]
    (OUT / "single_stage_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def _two_stage(det, clf, im, args, device):
    b, cf = yolo_detect(det, im, args.det_imgsz, args.conf, device)
    cr = [crop_box(im, x, args.margin) or im for x in b]
    return classify_pil_batch(clf, cr, device, args.img_size) if cr else None


if __name__ == "__main__":
    main()
