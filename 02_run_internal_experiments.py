"""
02_run_internal_experiments.py
Experimentos no dataset interno (imagens já recortadas), com split AGRUPADO + augmentation só no treino,
várias sementes e comparação no MESMO conjunto de teste de:

    direct_none / direct_uniform / direct_targeted : classificação direta (imagem inteira) x augmentation (R1.7)
    center_targeted                                : recorte central simples (80%)         (R1.6)
    yolo_targeted                                  : recorte pelo YOLOv8n (treinado só no TREINO) (R1.3, R1.6)
    direct_targeted_<arch>                         : baselines de classificação (R1.10, R3.4) [opcional]

Observações importantes
  * As "classes difíceis" NÃO são mais fixas: em cada semente são as K classes com menor probabilidade média da
    classe verdadeira na VALIDAÇÃO do modelo direct_none (o teste nunca participa)  -> responde a R1.7.
  * Se o YOLO não detectar nada (conf < --yolo-conf) usa-se a imagem inteira (fallback), então todos os pipelines
    são avaliados exatamente nas mesmas imagens de teste.
  * O script é RETOMÁVEL: resultados já salvos em results/internal/seed*/<config>.json são pulados (use --force).

Uso típico:
    python 02_run_internal_experiments.py --seeds 0                     # teste rápido (1 semente)
    python 02_run_internal_experiments.py --seeds 0 1 2 3 4             # execução completa
    python 02_run_internal_experiments.py --seeds 0 1 2 3 4 --configs direct_none direct_uniform direct_targeted center_targeted  # sem YOLO
    python 02_run_internal_experiments.py --aggregate-only              # só refaz as tabelas
"""
import argparse
import shutil
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import (ARCHS, RESULTS_DIR, ROOT, SPLITS_DIR, WORK_DIR, ArrayDataset, build_cache, build_model,
                    center_crop_pil, classification_metrics, classify_pil, crop_box, env_info, group_split,
                    list_dataset, load_json, load_rgb, make_transforms, pick_device, predict_probs, save_json,
                    seed_everything, time_calls, to_cache_size, train_classifier, train_yolo, write_yolo_yaml,
                    yolo_detect, yolo_val)

CONFIGS = {
    "direct_none": ("direct", "none", "resnet18"),
    "direct_uniform": ("direct", "uniform", "resnet18"),
    "direct_targeted": ("direct", "targeted", "resnet18"),
    "center_targeted": ("center", "targeted", "resnet18"),
    "yolo_targeted": ("yolo", "targeted", "resnet18"),
    "direct_targeted_resnet50": ("direct", "targeted", "resnet50"),
    "direct_targeted_mobilenet_v3_large": ("direct", "targeted", "mobilenet_v3_large"),
    "direct_targeted_efficientnet_b0": ("direct", "targeted", "efficientnet_b0"),
}
DEFAULT_CONFIGS = ["direct_none", "direct_uniform", "direct_targeted", "center_targeted", "yolo_targeted"]
OUT = RESULTS_DIR / "internal"


# ----------------------------------------------------------------------------
def load_groups(paths):
    df = pd.read_csv(SPLITS_DIR / "groups.csv")
    m = dict(zip(df["path"], df["group_id"]))
    missing = [p for p in paths if p not in m]
    if missing:
        raise SystemExit("groups.csv não corresponde ao dataset atual. Rode 01_build_groups.py de novo.")
    return np.array([m[p] for p in paths])


def yolo_stage(seed, paths, labels, split, args, device):
    """Treina o YOLO (caixa proxy = imagem inteira) SÓ com treino/val e gera recortes para TODAS as imagens."""
    from ultralytics import YOLO
    stage_json = OUT / f"seed{seed}" / "yolo_stage.json"
    crops_npy = ROOT / "cache" / f"yolo_crops_seed{seed}.npy"
    if stage_json.exists() and crops_npy.exists() and not args.force:
        return load_json(stage_json), np.load(crops_npy)

    tr, va, te = split
    root = WORK_DIR / f"yolo_proxy_seed{seed}"
    if root.exists():
        shutil.rmtree(root)
    for sp, idxs in (("train", tr), ("val", va)):
        (root / "images" / sp).mkdir(parents=True)
        (root / "labels" / sp).mkdir(parents=True)
        for i in idxs:
            img = load_rgb(ROOT / paths[i])
            s = 640 / max(img.size)
            if s < 1:
                img = img.resize((int(img.width * s), int(img.height * s)), Image.BICUBIC)
            img.save(root / "images" / sp / f"{i:04d}.jpg", quality=92)
            (root / "labels" / sp / f"{i:04d}.txt").write_text("0 0.5 0.5 1.0 1.0\n")
    yaml = root / "dataset.yaml"
    write_yolo_yaml(yaml, root)

    print(f"   [YOLO seed {seed}] treinando ({args.yolo_epochs} épocas, imgsz {args.yolo_imgsz}) ...")
    t0 = time.time()
    best = train_yolo(yaml, root / "runs", "train", args.yolo_weights, args.yolo_epochs, args.yolo_imgsz,
                      batch=16, seed=seed, device=device, workers=0, patience=10)
    train_s = time.time() - t0
    proxy = yolo_val(best, yaml, "val", args.yolo_imgsz, device, root / "runs", "val")

    model = YOLO(str(best))
    crops = np.zeros((len(paths), 256, 256, 3), dtype=np.uint8)
    found = np.zeros(len(paths), dtype=bool)
    iou_full = np.full(len(paths), np.nan)
    for i, p in enumerate(paths):
        img = load_rgb(ROOT / p)
        boxes, confs = yolo_detect(model, img, args.yolo_imgsz, args.yolo_conf, device)
        crop = None
        if len(boxes):
            j = int(np.argmax(confs))                    # regra: UMA caixa por imagem = a de maior confiança
            crop = crop_box(img, boxes[j])
            if crop is not None:
                found[i] = True
                x1, y1, x2, y2 = boxes[j]
                inter = max(0, min(x2, img.width) - max(x1, 0)) * max(0, min(y2, img.height) - max(y1, 0))
                iou_full[i] = inter / (img.width * img.height + (x2 - x1) * (y2 - y1) - inter)
        crops[i] = to_cache_size(crop if crop is not None else img)   # fallback: imagem inteira
    (ROOT / "cache").mkdir(exist_ok=True)
    np.save(crops_npy, crops)
    info = {
        "best_pt": str(best), "train_seconds": train_s, "conf_threshold": args.yolo_conf,
        "box_rule": "uma caixa por imagem: a de maior confiança com conf >= conf_threshold; sem detecção -> imagem inteira",
        "imgsz": args.yolo_imgsz, "epochs": args.yolo_epochs,
        "proxy_val_map50": proxy["map50"], "proxy_val_map50_95": proxy["map50_95"],
        "proxy_val_precision": proxy["precision"], "proxy_val_recall": proxy["recall"],
        "detection_rate": {k: float(found[ix].mean()) for k, ix in zip(("train", "val", "test"), (tr, va, te))},
        "mean_iou_with_full_image_box": {k: float(np.nanmean(iou_full[ix])) if np.isfinite(iou_full[ix]).any() else None
                                         for k, ix in zip(("train", "val", "test"), (tr, va, te))},
        "note": "mAP medido contra caixas-proxy (imagem inteira); NÃO é desempenho de localização em cenas reais.",
    }
    save_json(info, stage_json)
    return info, crops


def timing_fn(pipeline, model, device, args, yolo_pt=None):
    tf = make_transforms(args.img_size)["eval"]
    if pipeline == "direct":
        return (lambda im: classify_pil(model, im, device, tf=tf)), None
    if pipeline == "center":
        return (lambda im: classify_pil(model, center_crop_pil(im, args.center_ratio), device, tf=tf)), None
    from ultralytics import YOLO
    ym = YOLO(yolo_pt)

    def full(im):
        boxes, confs = yolo_detect(ym, im, args.yolo_imgsz, args.yolo_conf, device)
        crop = crop_box(im, boxes[int(np.argmax(confs))]) if len(boxes) else None
        return classify_pil(model, crop if crop is not None else im, device, tf=tf)

    det_only = lambda im: yolo_detect(ym, im, args.yolo_imgsz, args.yolo_conf, device)
    return full, det_only


def run_one(cfg_name, seed, images, labels, paths, split, weak, classes, args, device, yolo_info=None):
    pipeline, aug, arch = CONFIGS[cfg_name]
    tr, va, te = split
    seed_everything(seed)
    train_ds = ArrayDataset(images[tr], labels[tr], aug=aug, weak=weak, repeat=args.repeat, train=True,
                            img_size=args.img_size)
    val_ds = ArrayDataset(images[va], labels[va], img_size=args.img_size)
    test_ds = ArrayDataset(images[te], labels[te], img_size=args.img_size)
    print(f"   [{cfg_name}] treino {len(tr)} (+{train_ds.extra_samples_per_epoch()} amostras/época por repetição) | "
          f"val {len(va)} | teste {len(te)}")
    model = build_model(arch, len(classes), pretrained=not args.no_pretrained)
    model, hist = train_classifier(model, train_ds, val_ds, device, epochs=args.epochs, patience=args.patience,
                                   lr=args.lr, batch=args.batch)
    probs = predict_probs(model, test_ds, device)
    metrics = classification_metrics(labels[te], probs.argmax(1), classes)

    # tempo ponta-a-ponta por imagem (lote=1, imagem JÁ decodificada em memória; inclui detector + recorte + pré-proc. + rede)
    n_t = min(args.time_n, len(te))
    pil_imgs = [load_rgb(ROOT / paths[i]) for i in te[:n_t]]
    fn, det_only = timing_fn(pipeline, model, device, args, yolo_info["best_pt"] if yolo_info else None)
    timing = {"end_to_end": time_calls(fn, pil_imgs)}
    if det_only is not None:
        timing["detector_only"] = time_calls(det_only, pil_imgs)
    timing["protocol"] = (f"CPU/GPU={device}, lote=1, {n_t} imagens de teste, 3 chamadas de aquecimento descartadas, "
                          "imagem já decodificada (exclui leitura de disco); inclui detector+recorte+pré-processamento+rede")

    res = {
        "config": cfg_name, "seed": seed, "pipeline": pipeline, "augmentation": aug, "arch": arch,
        "hparams": {"img_size": args.img_size, "epochs_max": args.epochs, "patience": args.patience,
                    "lr": args.lr, "batch": args.batch, "repeat_weak": args.repeat, "pretrained": not args.no_pretrained},
        "split_sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "weak_classes": [classes[w] for w in weak] if aug == "targeted" else [],
        "extra_samples_per_epoch": int(train_ds.extra_samples_per_epoch()),
        "training": {k: v for k, v in hist.items() if k != "history"},
        "history": hist["history"],
        "metrics": metrics, "timing": timing, "yolo_stage": yolo_info, "env": env_info(),
    }
    save_json(res, OUT / f"seed{seed}" / f"{cfg_name}.json")
    print(f"   [{cfg_name}] seed {seed}: acc {metrics['accuracy']:.4f} | F1-macro {metrics['f1_macro']:.4f} | "
          f"{timing['end_to_end']['mean_ms']:.1f} ms/img")
    return res, model


# ----------------------------------------------------------------------------
def aggregate():
    rows, pcs, cms = [], [], {}
    for f in sorted(OUT.glob("seed*/*.json")):
        if f.name in ("yolo_stage.json", "weak_classes.json"):
            continue
        r = load_json(f)
        m = r["metrics"]
        rows.append({"config": r["config"], "seed": r["seed"], "accuracy": m["accuracy"],
                     "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
                     "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"],
                     "time_ms": r["timing"]["end_to_end"]["mean_ms"], "n_test": m["n"],
                     "best_epoch": r["training"]["best_epoch"]})
        for c, v in m["per_class"].items():
            pcs.append({"config": r["config"], "seed": r["seed"], "class": c, "f1": v["f1"], "recall": v["recall"]})
        cms.setdefault(r["config"], []).append(np.array(m["confusion_matrix"]))
    if not rows:
        print("Nenhum resultado encontrado em", OUT)
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "internal_runs.csv", index=False)
    order = [c for c in CONFIGS if c in set(df["config"])]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "time_ms"]
    g = df.groupby("config")[metrics]
    mean, std = g.mean().reindex(order), g.std(ddof=1).reindex(order).fillna(0.0)
    summ = mean.copy()
    summ.columns = [f"{c}_mean" for c in mean.columns]
    for c in metrics:
        summ[f"{c}_std"] = std[c]
    summ["n_seeds"] = df.groupby("config")["seed"].nunique().reindex(order)
    summ.to_csv(OUT / "internal_summary.csv")

    def fmt(c, k):
        if k == "time_ms":
            return f"{mean.loc[c, k]:.1f} ± {std.loc[c, k]:.1f}"
        return f"{100 * mean.loc[c, k]:.2f} ± {100 * std.loc[c, k]:.2f}"

    md = ["| Config | n seeds | Accuracy (%) | Precision macro (%) | Recall macro (%) | F1 macro (%) | Time (ms/img) |",
          "|---|---|---|---|---|---|---|"]
    for c in order:
        md.append(f"| {c} | {int(summ.loc[c, 'n_seeds'])} | " + " | ".join(fmt(c, k) for k in metrics) + " |")

    # diferenças pareadas (mesma semente = mesmo teste) em relação ao baseline direct_targeted
    if "direct_targeted" in order:
        base = df[df.config == "direct_targeted"].set_index("seed")
        md += ["", "Diferença pareada em relação a direct_targeted (pontos percentuais, média ± dp entre sementes):", "",
               "| Config | Δ Accuracy | Δ F1 macro |", "|---|---|---|"]
        for c in order:
            if c == "direct_targeted":
                continue
            d = df[df.config == c].set_index("seed")
            common = d.index.intersection(base.index)
            if len(common) == 0:
                continue
            da = 100 * (d.loc[common, "accuracy"] - base.loc[common, "accuracy"])
            dfm = 100 * (d.loc[common, "f1_macro"] - base.loc[common, "f1_macro"])
            md.append(f"| {c} | {da.mean():+.2f} ± {da.std(ddof=1) if len(common) > 1 else 0:.2f} | "
                      f"{dfm.mean():+.2f} ± {dfm.std(ddof=1) if len(common) > 1 else 0:.2f} |")
    (OUT / "internal_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    pd.DataFrame(pcs).pivot_table(index="class", columns="config", values="f1", aggfunc="mean") \
        .to_csv(OUT / "internal_per_class_f1.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        classes = get_classes_from_json()
        for cfg in ("direct_targeted", "yolo_targeted"):
            if cfg in cms:
                cm = np.sum(cms[cfg], axis=0)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cbar=False)
                plt.title(f"Confusion matrix – {cfg} (sum over {len(cms[cfg])} seeds, test subsets)")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.savefig(OUT / f"confusion_{cfg}.png", dpi=200)
                plt.close()
    except Exception as e:  # gráfico é opcional
        print("(gráfico da matriz de confusão não gerado:", e, ")")
    print(f"\nTabelas salvas em {OUT}")


def get_classes_from_json():
    from common import get_classes
    return get_classes()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--weak-k", type=int, default=5, help="nº de classes 'difíceis' escolhidas na validação")
    ap.add_argument("--repeat", type=int, default=3, help="repetições/época das classes fracas em 'targeted'")
    ap.add_argument("--center-ratio", type=float, default=0.8)
    ap.add_argument("--yolo-epochs", type=int, default=15)
    ap.add_argument("--yolo-imgsz", type=int, default=320)
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--time-n", type=int, default=100)
    ap.add_argument("--no-pretrained", action="store_true", help="(apenas teste de fumaça)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate()
        return

    device = pick_device()
    print("Dispositivo:", device, "| threads:", torch.get_num_threads())
    items, classes = list_dataset()
    paths = [p for p, _ in items]
    labels = np.array([y for _, y in items])
    groups = load_groups(paths)
    configs = ["direct_none"] + [c for c in args.configs if c != "direct_none"]   # direct_none é sempre rodado primeiro

    caches = {}
    if any(CONFIGS[c][0] == "direct" for c in configs):
        caches["direct"] = build_cache(paths, lambda im: im, "direct_256")
    if any(CONFIGS[c][0] == "center" for c in configs):
        caches["center"] = build_cache(paths, lambda im: center_crop_pil(im, args.center_ratio),
                                       f"center{int(args.center_ratio * 100)}_256")

    for seed in args.seeds:
        print(f"\n================ SEMENTE {seed} ================")
        split = group_split(labels, groups, seed)
        tr, va, te = split
        print(f"split agrupado: treino {len(tr)} | val {len(va)} | teste {len(te)} "
              f"| classes ausentes no teste: {sorted(set(range(len(classes))) - set(labels[te].tolist()))}")
        pd.DataFrame({"path": paths, "class": [classes[y] for y in labels], "group_id": groups,
                      "split": np.select([np.isin(np.arange(len(paths)), tr), np.isin(np.arange(len(paths)), va)],
                                         ["train", "val"], "test")}).to_csv(SPLITS_DIR / f"seed{seed}_split.csv", index=False)

        weak_file = OUT / f"seed{seed}" / "weak_classes.json"
        weak = None
        yolo_info, yolo_crops = None, None
        for cfg in configs:
            out_file = OUT / f"seed{seed}" / f"{cfg}.json"
            pipeline, aug, arch = CONFIGS[cfg]
            if out_file.exists() and not args.force:
                print(f"   [{cfg}] já existe, pulando.")
                if cfg == "direct_none" and weak_file.exists():
                    weak = [classes.index(c) for c in load_json(weak_file)["weak_classes"]]
                continue
            if aug == "targeted" and weak is None:
                if weak_file.exists():
                    weak = [classes.index(c) for c in load_json(weak_file)["weak_classes"]]
                else:
                    raise SystemExit("rode direct_none antes (define as classes fracas pela validação).")
            if pipeline == "yolo":
                yolo_info, yolo_crops = yolo_stage(seed, paths, labels, split, args, device)
                images = yolo_crops
            else:
                images = caches[pipeline]
            res, model = run_one(cfg, seed, images, labels, paths, split, weak or [], classes, args, device,
                                 yolo_info if pipeline == "yolo" else None)
            if cfg == "direct_none":
                weak = select_weak(model, images, labels, va, classes, args, device)
                save_json({"seed": seed, "weak_classes": [classes[w] for w in weak],
                           "rule": f"{args.weak_k} classes com menor probabilidade média da classe verdadeira "
                                   f"na VALIDAÇÃO do modelo direct_none (o teste não participa)"}, weak_file)
                print(f"   classes fracas (semente {seed}): {[classes[w] for w in weak]}")
    aggregate()


def select_weak(model, images, labels, va, classes, args, device):
    """K classes com menor probabilidade média da classe verdadeira na VALIDAÇÃO (critério contínuo, sem empates)."""
    probs = predict_probs(model, ArrayDataset(images[va], labels[va], img_size=args.img_size), device)
    p_true = probs[np.arange(len(va)), labels[va]]
    score = {c: float(p_true[labels[va] == c].mean()) for c in range(len(classes)) if (labels[va] == c).any()}
    return sorted(score, key=lambda c: (score[c], c))[:args.weak_k]


if __name__ == "__main__":
    main()
