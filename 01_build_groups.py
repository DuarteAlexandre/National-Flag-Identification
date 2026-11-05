"""
01_build_groups.py
Agrupa imagens quase-duplicadas do dataset interno. Como NÃO existem registros de fonte por imagem
(nem os vídeos originais), este agrupamento é o melhor substituto possível para "split por fonte":
imagens que são (quase) a mesma imagem passam a ficar SEMPRE no mesmo subconjunto (treino/val/teste).

Regra (dentro da mesma classe, calibrada visualmente no dataset):
    duas imagens são unidas se  MD5 igual   OU   (distância de Hamming dHash-256 <= HAM_THR
                                                   E correlação de miniaturas 16x16 >= COS_THR)
Grupos = componentes conexos.

Saídas (pasta splits/):
    groups.csv             path, class, group_id, group_size, md5
    cluster_report.txt     estatísticas para o texto do artigo (resposta a R1.2 / R1.5)
    cluster_examples.jpg   montagem dos maiores grupos, PARA CONFERÊNCIA VISUAL

Uso:   python 01_build_groups.py
"""
import argparse
import csv
import hashlib

import numpy as np
from PIL import Image, ImageDraw

from common import ROOT, SPLITS_DIR, list_dataset


def descriptors(path):
    im = Image.open(path).convert("RGB")
    g = np.asarray(im.convert("L").resize((17, 16), Image.BICUBIC), dtype=np.float32)
    dh = (g[:, 1:] > g[:, :-1]).flatten()                       # 256 bits
    t = np.asarray(im.resize((16, 16), Image.BICUBIC), dtype=np.float32).flatten()
    t = t - t.mean()
    t = t / (np.linalg.norm(t) + 1e-6)
    return dh, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham-thr", type=int, default=24)
    ap.add_argument("--cos-thr", type=float, default=0.90)
    args = ap.parse_args()

    items, classes = list_dataset()
    paths = [p for p, _ in items]
    labels = np.array([y for _, y in items])
    print(f"{len(paths)} imagens, {len(classes)} classes")

    md5, dh, th = [], [], []
    for i, p in enumerate(paths):
        md5.append(hashlib.md5((ROOT / p).read_bytes()).hexdigest())
        a, b = descriptors(ROOT / p)
        dh.append(a)
        th.append(b)
        if (i + 1) % 250 == 0:
            print(f"   descritores {i + 1}/{len(paths)}")
    dh, th, md5 = np.array(dh), np.array(th), np.array(md5)

    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_md5_pairs = 0
    for c in range(len(classes)):
        idx = np.where(labels == c)[0]
        H = (dh[idx][:, None, :] != dh[idx][None, :, :]).sum(2)
        C = th[idx] @ th[idx].T
        same = md5[idx][:, None] == md5[idx][None, :]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if same[a, b]:
                    n_md5_pairs += 1
                if same[a, b] or (H[a, b] <= args.ham_thr and C[a, b] >= args.cos_thr):
                    parent[find(idx[a])] = find(idx[b])

    root = np.array([find(i) for i in range(len(paths))])
    uniq = {r: k for k, r in enumerate(sorted(set(root.tolist())))}
    gid = np.array([uniq[r] for r in root])
    sizes = np.bincount(gid)

    SPLITS_DIR.mkdir(exist_ok=True)
    with open(SPLITS_DIR / "groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "group_id", "group_size", "md5"])
        for i, p in enumerate(paths):
            w.writerow([p, classes[labels[i]], int(gid[i]), int(sizes[gid[i]]), md5[i]])

    # ---- relatório
    multi = int((sizes[gid] > 1).sum())
    lines = [
        f"Imagens: {len(paths)} | classes: {len(classes)}",
        f"Parâmetros: Hamming(dHash-256) <= {args.ham_thr} E correlação(miniatura 16x16) >= {args.cos_thr}, ou MD5 idêntico",
        f"Pares com arquivo idêntico (MD5): {n_md5_pairs}",
        f"Grupos: {len(sizes)} | imagens pertencentes a grupos com >1 imagem: {multi} ({100 * multi / len(paths):.1f}%)",
        f"Distribuição do tamanho dos grupos: " + ", ".join(f"{s}:{int((sizes == s).sum())}" for s in sorted(set(sizes.tolist()))),
        "",
        "Por classe: imagens / grupos / maior grupo",
    ]
    for c, name in enumerate(classes):
        g = gid[labels == c]
        cnt = np.bincount(g)[np.bincount(g) > 0]
        lines.append(f"  {name:<12} {len(g):>3} / {len(cnt):>3} / {cnt.max():>2}")
    (SPLITS_DIR / "cluster_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ---- montagem dos 10 maiores grupos
    T = 110
    top = np.argsort(-sizes)[:10]
    sheet = Image.new("RGB", (8 * T, len(top) * T), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for r, g in enumerate(top):
        mem = np.where(gid == g)[0][:8]
        for k, i in enumerate(mem):
            im = Image.open(ROOT / paths[i]).convert("RGB")
            im.thumbnail((T - 4, T - 4))
            sheet.paste(im, (k * T + 2, r * T + 2))
        d.text((3, r * T + 3), f"grupo {g} (n={sizes[g]}) {classes[labels[mem[0]]]}", fill=(255, 255, 0))
    sheet.save(SPLITS_DIR / "cluster_examples.jpg", quality=85)
    print(f"\nSalvo em {SPLITS_DIR}. CONFIRA visualmente splits/cluster_examples.jpg "
          f"(cada linha deve ser a 'mesma' imagem/arte).")


if __name__ == "__main__":
    main()
