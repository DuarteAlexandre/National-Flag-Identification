"""
12_prepare_maritime.py
Valida as anotações exportadas (formato YOLO), faz o split POR VÍDEO (treino/val/teste), monta o dataset do
detector de bandeiras (classe única "flag") e gera estatísticas + planilha de atributos do teste.

Pré-requisitos:
  maritime/frames/<video_id>_<frame>.jpg     (gerados por 10_extract_frames.py)
  maritime/labels/<video_id>_<frame>.txt     (exportação YOLO da ferramenta de anotação; 1 .txt POR FRAME,
                                              vazio = frame SEM bandeira / negativo)
  maritime/classes.txt                       (ordem das classes usada na anotação; já fornecido)

Uso:
    python 12_prepare_maritime.py                       # split aleatório por vídeo (semente 0)
    python 12_prepare_maritime.py --test-videos v03 v08 # força quais vídeos vão para o TESTE
    python 12_prepare_maritime.py --resplit             # refaz o split (só ANTES de rodar os experimentos!)

IMPORTANTE: depois de rodar os experimentos, NÃO refaça o split (maritime/split.csv congela o teste).
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from common import MARITIME_DIR, ROOT, WORK_DIR, crop_box, write_yolo_yaml

FRAMES, LABELS = MARITIME_DIR / "frames", MARITIME_DIR / "labels"
YOLO_DIR = MARITIME_DIR / "yolo_flag"


def size_bucket(bw_px, bh_px):
    s = (bw_px * bh_px) ** 0.5
    return "small" if s < 32 else ("medium" if s < 96 else "large")   # critério análogo ao COCO (frames de 1280 px de largura)


def read_labels(stem, n_classes):
    f = LABELS / f"{stem}.txt"
    if not f.exists():
        return None
    boxes = []
    for ln, line in enumerate(f.read_text().splitlines()):
        if not line.strip():
            continue
        p = line.split()
        c, x, y, w, h = int(p[0]), *map(float, p[1:5])
        if not (0 <= c < n_classes) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise SystemExit(f"Anotação inválida em {f.name}, linha {ln + 1}: '{line}'")
        boxes.append((c, x, y, w, h))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.6, 0.15, 0.25], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--test-videos", nargs="*", default=None)
    ap.add_argument("--val-videos", nargs="*", default=None)
    ap.add_argument("--unit", choices=["video", "image"], default="video",
                    help="'video': split por vídeo (frames do mesmo vídeo juntos). 'image': cada foto é independente "
                         "(fotos coletadas manualmente), split estratificado por país. Ex.: --unit image --fractions 0.5 0.1 0.4")
    ap.add_argument("--resplit", action="store_true")
    ap.add_argument("--missing-as-negative", action="store_true",
                    help="frames sem .txt são tratados como SEM bandeira (use só se TODOS os frames foram revisados)")
    args = ap.parse_args()

    names = [l.strip() for l in (MARITIME_DIR / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert names[-1] == "other", "a última classe de maritime/classes.txt deve ser 'other'"
    n_cls = len(names)

    frames = sorted(p for p in FRAMES.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"Sem frames em {FRAMES}")
    LABELS.mkdir(exist_ok=True)
    ann, missing = {}, []
    for p in frames:
        b = read_labels(p.stem, n_cls)
        (missing.append(p.name) if b is None else ann.__setitem__(p.stem, b))
    if missing and args.missing_as_negative:
        for name in missing:                       # frame revisado e SEM bandeira -> cria .txt vazio
            (LABELS / (Path(name).stem + ".txt")).write_text("")
            ann[Path(name).stem] = []
        print(f"{len(missing)} frames sem arquivo de rótulo foram tratados como NEGATIVOS (.txt vazio criado).")
        missing = []
    if missing:
        print(f"ATENÇÃO: {len(missing)} frames SEM arquivo de rótulo (serão IGNORADOS). Exemplos: {missing[:5]}")
        print("         Se TODOS os frames foram revisados e esses não têm bandeira, rode com --missing-as-negative")
    stems = sorted(ann)
    vid_of = {s: s.rsplit("_", 1)[0] for s in stems}
    videos = sorted(set(vid_of.values()))
    print(f"{len(stems)} frames anotados, {len(videos)} vídeos")

    # ---------------- split por vídeo (congelado em split.csv)
    split_csv = MARITIME_DIR / "split.csv"
    if split_csv.exists() and not args.resplit:
        split_of_video = {r["video_id"]: r["split"] for r in csv.DictReader(open(split_csv, encoding="utf-8"))}
        new = [v for v in videos if v not in split_of_video]
        if new:
            raise SystemExit(f"Vídeos novos sem split: {new}. Use --resplit (somente antes dos experimentos).")
        print("Usando split congelado em", split_csv)
    else:
        rng = np.random.default_rng(args.seed)
        if args.unit == "image":
            # cada foto é uma unidade independente; split ESTRATIFICADO pelo país da 1ª caixa ("neg" = sem bandeira)
            strata = {}
            for s in stems:
                key = names[ann[s][0][0]] if ann[s] else "neg"
                strata.setdefault(key, []).append(vid_of[s])
            split_of_video = {}
            ftr, fva, fte = args.fractions
            for key, vs in sorted(strata.items()):
                vs = sorted(set(vs))
                rng.shuffle(vs)
                n = len(vs)
                n_te = max(1, round(fte * n)) if n >= 3 else 0
                n_va = max(1, round(fva * n)) if n >= 5 else 0
                for i, v in enumerate(vs):
                    split_of_video[v] = "test" if i < n_te else ("val" if i < n_te + n_va else "train")
            print("Split por IMAGEM, estratificado por país (cada foto = 1 unidade).")
        else:
            forced = {v: "test" for v in (args.test_videos or [])}
            forced.update({v: "val" for v in (args.val_videos or [])})
            rest = [v for v in videos if v not in forced]
            rng.shuffle(rest)
            n_frames = {v: sum(1 for s in stems if vid_of[s] == v) for v in videos}
            total = sum(n_frames.values())
            target = {"train": args.fractions[0] * total, "val": args.fractions[1] * total, "test": args.fractions[2] * total}
            got = {k: sum(n_frames[v] for v, s in forced.items() if s == k) for k in target}
            split_of_video = dict(forced)
            for v in rest:       # atribui ao subconjunto mais "defasado" em relação à meta
                k = max(target, key=lambda k: target[k] - got[k])
                split_of_video[v] = k
                got[k] += n_frames[v]
        with open(split_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "split"])
            for v in videos:
                w.writerow([v, split_of_video[v]])
        print("Split por vídeo salvo em", split_csv)

    # ---------------- dataset do detector (classe única)
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for sp in ("train", "val", "test"):
        (YOLO_DIR / "images" / sp).mkdir(parents=True)
        (YOLO_DIR / "labels" / sp).mkdir(parents=True)
    stats = {sp: {"videos": set(), "frames": 0, "neg": 0, "boxes": 0, "per_class": np.zeros(n_cls, int),
                  "buckets": {"small": 0, "medium": 0, "large": 0}} for sp in ("train", "val", "test")}
    attr_rows = []
    (MARITIME_DIR / "attr_crops").mkdir(exist_ok=True)
    for s in stems:
        sp = split_of_video[vid_of[s]]
        shutil.copy2(FRAMES / f"{s}.jpg", YOLO_DIR / "images" / sp / f"{s}.jpg")
        boxes = ann[s]
        (YOLO_DIR / "labels" / sp / f"{s}.txt").write_text(
            "".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for _, x, y, w, h in boxes))
        W, H = Image.open(FRAMES / f"{s}.jpg").size
        st = stats[sp]
        st["videos"].add(vid_of[s])
        st["frames"] += 1
        st["neg"] += int(len(boxes) == 0)
        for line_no, (c, x, y, w, h) in enumerate(boxes):
            st["boxes"] += 1
            st["per_class"][c] += 1
            b = size_bucket(w * W, h * H)
            st["buckets"][b] += 1
            if sp == "test":
                rid = f"t{len(attr_rows) + 1:04d}"
                attr_rows.append([rid, s, line_no, names[c], b, "", "", ""])
                crop = crop_box(Image.open(FRAMES / f"{s}.jpg").convert("RGB"),
                                [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H], margin=0.1)
                if crop is not None:
                    sc = max(1.0, 128 / min(crop.size))
                    crop.resize((int(crop.width * sc), int(crop.height * sc)), Image.BICUBIC).save(
                        MARITIME_DIR / "attr_crops" / f"{rid}.jpg", quality=92)
    write_yolo_yaml(YOLO_DIR / "dataset.yaml", YOLO_DIR, names=("flag",), test="images/test")

    # ---------------- estatísticas
    md = ["| Subconjunto | Vídeos | Frames | Frames sem bandeira | Caixas | small | medium | large |", "|---|---|---|---|---|---|---|---|"]
    for sp, st in stats.items():
        md.append(f"| {sp} | {len(st['videos'])} | {st['frames']} | {st['neg']} | {st['boxes']} | "
                  f"{st['buckets']['small']} | {st['buckets']['medium']} | {st['buckets']['large']} |")
    md += ["", "Caixas por país:", "", "| País | train | val | test |", "|---|---|---|---|"]
    for i, n in enumerate(names):
        md.append(f"| {n} | {stats['train']['per_class'][i]} | {stats['val']['per_class'][i]} | {stats['test']['per_class'][i]} |")
    md += ["", "Vídeos por subconjunto:"] + [f"- {sp}: {', '.join(sorted(st['videos']))}" for sp, st in stats.items()]
    (MARITIME_DIR / "stats.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    with open(MARITIME_DIR / "attributes_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "line", "country", "size_bucket", "blurred", "occluded", "deformed"])
        w.writerows(attr_rows)

    for sp in ("train", "val", "test"):
        if stats[sp]["frames"] == 0 or stats[sp]["boxes"] == 0:
            print(f"\n*** ALERTA: subconjunto '{sp}' sem frames ou sem caixas. Refaça o split (--resplit, outra --seed "
                  f"ou --test-videos) ANTES de rodar os experimentos.")
    print(f"\nDataset do detector em {YOLO_DIR}\nPlanilha de atributos (teste): maritime/attributes_template.csv "
          f"(miniaturas em maritime/attr_crops/)")


if __name__ == "__main__":
    main()
