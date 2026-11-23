"""
14_compare_classifiers_maritime.py
Comparação ResNet18 x MobileNetV3-Large nos MESMOS recortes de um detector FIXO (resposta a R1.4, 2a rodada).

O que faz
  1. Reaproveita o detector marítimo já treinado (YOLOv8n, classe única) e o classificador B (ResNet18) já usado nas
     Tabelas 9-10 (results/maritime/classifier_B.pt). Nada disso é retreinado.
  2. Treina um classificador B equivalente com MobileNetV3-Large: mesma receita do classificador B
     (treino + teste internos da semente 0 como treino, validação interna para parada antecipada, recortes das caixas
     marítimas de treino/validação adicionados, augmentation 'targeted' com as mesmas classes fracas, 30 épocas,
     paciência 7, lote 32, lr 3e-4, entrada 224x224).
  3. No TESTE marítimo, usando as MESMAS detecções (mesmas caixas e mesmos recortes) para os dois modelos:
     acurácia "oráculo" (caixa anotada -> classificador), métricas ponta a ponta no limiar 0.25, acurácia top-1 por frame,
     diferenças pareadas (IC bootstrap 95%) e contagem de discordâncias.
  4. Mede latência no MESMO computador, com lote 1 e frame já decodificado: (a) só o classificador, por recorte;
     (b) pipeline completo por frame (detector + recorte + classificador de todas as caixas). Execuções alternadas
     (A B A B ...) para reduzir o efeito de variações do computador.

Uso (na pasta do projeto, onde estão common.py e 13_run_maritime_experiments.py):
    python 14_compare_classifiers_maritime.py
    python 14_compare_classifiers_maritime.py --epochs 1        # teste de fumaça
Saídas: results/maritime/compare_classifiers.json, results/maritime/compare_classifiers.md,
        results/maritime/classifier_B_mobilenet_v3_large.pt
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common as C
from common import (ArrayDataset, MARITIME_DIR, RESULTS_DIR, ROOT, WORK_DIR, build_cache, build_model, classify_pil_batch,
                    classification_metrics, crop_box, env_info, group_split, list_dataset, load_rgb, pick_device, save_json,
                    seed_everything, train_classifier, yolo_detect)

_spec = importlib.util.spec_from_file_location("m13", Path(__file__).resolve().parent / "13_run_maritime_experiments.py")
m13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m13)          # main() de 13 só roda com __main__

OUT = RESULTS_DIR / "maritime"


def load_resnet_B(classes, device):
    m = build_model("resnet18", len(classes), pretrained=False)
    m.load_state_dict(torch.load(OUT / "classifier_B.pt", map_location="cpu"))
    return m.to(device).eval()


def get_mobilenet_B(classes, args, device, int_imgs, int_lbls, groups, mar_tr, mar_va):
    f = OUT / "classifier_B_mobilenet_v3_large.pt"
    model = build_model("mobilenet_v3_large", len(classes), pretrained=not args.no_pretrained)
    if f.exists() and not args.force_retrain:
        model.load_state_dict(torch.load(f, map_location="cpu"))
        return model.to(device).eval()
    seed_everything(args.seed)
    tr, va, te = group_split(int_lbls, groups, seed=0)
    tr = np.concatenate([tr, te])                    # igual ao classificador B do script 13
    Xtr, ytr, Xva, yva = int_imgs[tr], int_lbls[tr], int_imgs[va], int_lbls[va]
    Xtr, ytr = np.concatenate([Xtr, mar_tr[0]]), np.concatenate([ytr, mar_tr[1]])
    Xva, yva = np.concatenate([Xva, mar_va[0]]), np.concatenate([yva, mar_va[1]])
    weak = m13.final_weak(classes, args.weak_k)
    print(f"[MobileNetV3-L] treino {len(ytr)} | val {len(yva)} | classes fracas: {[classes[w] for w in weak]}")
    tds = ArrayDataset(Xtr, ytr, aug="targeted", weak=weak, repeat=args.repeat, train=True, img_size=args.img_size)
    vds = ArrayDataset(Xva, yva, img_size=args.img_size)
    model, hist = train_classifier(model, tds, vds, device, epochs=args.epochs, patience=args.patience, lr=args.lr,
                                   batch=args.batch)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f)
    save_json(hist | {"weak_classes": [classes[w] for w in weak]}, OUT / "classifier_B_mobilenet_v3_large_training.json")
    return model.eval()


def boot_diff(a, b, n=10000, seed=0):
    """IC bootstrap 95% da diferença de médias (b - a) para vetores 0/1 pareados."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def timing(models, ymodel, frames_pil, gt_crops_pil, args, device):
    """Execuções alternadas; devolve estatísticas por modelo (ms)."""
    res = {k: {"clf_only": [], "pipeline": [], "pipeline_pass_means": []} for k in models}
    for k, m in models.items():                      # aquecimento
        for im in frames_pil[:3]:
            b, _ = yolo_detect(ymodel, im, args.det_imgsz, args.conf, device)
        classify_pil_batch(m, gt_crops_pil[:1], device, args.img_size)
    for _ in range(args.timing_passes):
        for k, m in models.items():
            ts = []
            for im in gt_crops_pil:
                t0 = time.perf_counter()
                classify_pil_batch(m, [im], device, args.img_size)
                ts.append((time.perf_counter() - t0) * 1000)
            res[k]["clf_only"] += ts
            ts = []
            for im in frames_pil:
                t0 = time.perf_counter()
                b, cf = yolo_detect(ymodel, im, args.det_imgsz, args.conf, device)
                cr = [crop_box(im, x, args.margin) or im for x in b]
                if cr:
                    classify_pil_batch(m, cr, device, args.img_size)
                ts.append((time.perf_counter() - t0) * 1000)
            res[k]["pipeline"] += ts
            res[k]["pipeline_pass_means"].append(float(np.mean(ts)))
    out = {}
    for k, d in res.items():
        out[k] = {"clf_only_ms_mean": float(np.mean(d["clf_only"])), "clf_only_ms_sd": float(np.std(d["clf_only"])),
                  "clf_only_ms_median": float(np.median(d["clf_only"])),
                  "pipeline_ms_mean": float(np.mean(d["pipeline"])), "pipeline_ms_sd": float(np.std(d["pipeline"])),
                  "pipeline_ms_median": float(np.median(d["pipeline"])),
                  "pipeline_pass_means_ms": d["pipeline_pass_means"], "n_crops": len(gt_crops_pil), "n_frames": len(frames_pil)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--det-imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--weak-k", type=int, default=5)
    ap.add_argument("--timing-passes", type=int, default=3)
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="(apenas teste de fumaça)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    items, classes = list_dataset()
    paths = [p for p, _ in items]
    int_lbls = np.array([y for _, y in items])
    groups = pd.read_csv(ROOT / "splits" / "groups.csv").set_index("path").loc[paths, "group_id"].values
    int_imgs = build_cache(paths, lambda im: im, "direct_256")

    fr_train, fr_val, fr_test = (m13.load_frames(s, len(classes)) for s in ("train", "val", "test"))
    mar_tr, mar_va = m13.gt_crops(fr_train, args.margin), m13.gt_crops(fr_val, args.margin)

    models = {"resnet18": load_resnet_B(classes, device),
              "mobilenet_v3_large": get_mobilenet_B(classes, args, device, int_imgs, int_lbls, groups, mar_tr, mar_va)}
    n_params = {k: int(sum(p.numel() for p in m.parameters())) for k, m in models.items()}

    # detector fixo
    from ultralytics import YOLO
    best = WORK_DIR / f"maritime_detector_seed{args.seed}" / "runs" / "train" / "weights" / "best.pt"
    ymodel = YOLO(str(best))

    # detecções únicas (as mesmas para os dois classificadores)
    cache, pil = {}, {}
    for fr in fr_test:
        im = load_rgb(fr["path"])
        pil[fr["stem"]] = im
        boxes, confs = yolo_detect(ymodel, im, args.det_imgsz, args.conf, device)
        crops = [crop_box(im, b, args.margin) or im for b in boxes]
        gt_c = [crop_box(im, b["xyxy"], args.margin) or im for b in fr["boxes"]]
        e = {"boxes": boxes, "confs": confs, "pred_cls": {}, "oracle": {}}
        for k, m in models.items():
            pr = classify_pil_batch(m, crops, device, args.img_size) if crops else np.zeros((0, len(classes)))
            e["pred_cls"][k] = pr.argmax(1) if len(pr) else np.zeros(0, dtype=int)
            po = classify_pil_batch(m, gt_c, device, args.img_size) if gt_c else np.zeros((0, len(classes)))
            e["oracle"][k] = po.argmax(1) if len(po) else np.zeros(0, dtype=int)
        cache[fr["stem"]] = e

    results = {"env": env_info(), "args": vars(args), "n_params": n_params, "n_test_frames": len(fr_test),
               "n_test_boxes": sum(len(f["boxes"]) for f in fr_test), "per_model": {}}
    per_box = {}
    for k in models:
        r, rows, _ = m13.evaluate(fr_test, cache, k, args.conf, classes)
        yt, yp = [], []
        for fr in fr_test:
            for g, b in enumerate(fr["boxes"]):
                if b["cls"] >= 0:
                    yt.append(b["cls"])
                    yp.append(int(cache[fr["stem"]]["oracle"][k][g]))
        orc = classification_metrics(yt, yp, classes, present_only=True)
        base = []
        for fr in fr_test:
            gts = {b["cls"] for b in fr["boxes"] if b["cls"] >= 0}
            if not gts:
                continue
            e = cache[fr["stem"]]
            sel = np.where(e["confs"] >= args.conf)[0]
            base.append(bool(len(sel)) and int(e["pred_cls"][k][sel[int(np.argmax(e["confs"][sel]))]]) in gts)
        df = pd.DataFrame(rows)
        known = df[df["country"] != "other"]
        per_box[k] = {"oracle": (np.array(yt) == np.array(yp)).astype(int), "e2e": known["e2e_correct"].astype(int).values,
                      "frame_top1": np.array(base, int)}
        results["per_model"][k] = {"operating": r, "oracle_accuracy": orc["accuracy"], "oracle_f1_macro": orc["f1_macro"],
                                   "oracle_n": orc["n"], "frame_top1_detected_box": float(np.mean(base))}
    a, b = "resnet18", "mobilenet_v3_large"
    results["paired_mobilenet_minus_resnet"] = {}
    for key in ("oracle", "e2e", "frame_top1"):
        d, lo, hi = boot_diff(per_box[a][key], per_box[b][key])
        b01 = int(((per_box[a][key] == 0) & (per_box[b][key] == 1)).sum())      # só MobileNet acerta
        b10 = int(((per_box[a][key] == 1) & (per_box[b][key] == 0)).sum())      # só ResNet acerta
        results["paired_mobilenet_minus_resnet"][key] = {"diff": d, "ci95": [lo, hi], "only_mobilenet_correct": b01,
                                                         "only_resnet_correct": b10, "n": int(len(per_box[a][key]))}

    # latência
    ims = [pil[f["stem"]] for f in fr_test]
    gt_crops_pil = [crop_box(pil[f["stem"]], b["xyxy"], args.margin) or pil[f["stem"]] for f in fr_test for b in f["boxes"]]
    results["timing"] = timing(models, ymodel, ims, gt_crops_pil, args, device)
    results["timing_protocol"] = (f"dispositivo={device}, lote=1, frame ja decodificado, imgsz={args.det_imgsz}, conf={args.conf}; "
                                  f"{args.timing_passes} passagens alternadas; 'pipeline' = detector + NMS + recortes + classificacao "
                                  f"de todas as caixas; 'clf_only' = pre-processamento + classificador por recorte anotado")
    save_json(results, OUT / "compare_classifiers.json")

    md = [f"Teste marítimo: {results['n_test_frames']} frames, {results['n_test_boxes']} caixas. Detector fixo; limiar {args.conf}.", "",
          "| Modelo | Params (M) | Oráculo acc | Oráculo F1 | Det. P / R | E2E P / R | Top-1 por frame | Classif. ms/recorte | Pipeline ms/frame |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in models:
        pm, o, t = results["per_model"][k], results["per_model"][k]["operating"], results["timing"][k]
        md.append(f"| {k} | {n_params[k] / 1e6:.2f} | {pm['oracle_accuracy']:.3f} | {pm['oracle_f1_macro']:.3f} | "
                  f"{o['det_precision']:.3f} / {o['det_recall']:.3f} | {o['e2e_precision']:.3f} / {o['e2e_recall']:.3f} | "
                  f"{pm['frame_top1_detected_box']:.3f} | {t['clf_only_ms_mean']:.1f} ± {t['clf_only_ms_sd']:.1f} "
                  f"(med {t['clf_only_ms_median']:.1f}) | {t['pipeline_ms_mean']:.1f} ± {t['pipeline_ms_sd']:.1f} "
                  f"(med {t['pipeline_ms_median']:.1f}) |")
    md += ["", "Diferenças pareadas (MobileNetV3-L − ResNet18), IC bootstrap 95% e discordâncias:", "",
           "| Medida | Diferença | IC 95% | só MobileNet acerta | só ResNet acerta | n |", "|---|---|---|---|---|---|"]
    for key, v in results["paired_mobilenet_minus_resnet"].items():
        md.append(f"| {key} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['only_mobilenet_correct']} | "
                  f"{v['only_resnet_correct']} | {v['n']} |")
    md += ["", results["timing_protocol"]]
    (OUT / "compare_classifiers.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
