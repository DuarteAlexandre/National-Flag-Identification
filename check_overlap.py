"""check_overlap.py
Verifica se alguma foto maritima (maritime/frames) e' duplicata ou quase-duplicata
de alguma imagem do dataset interno (dataset/). Usa os mesmos criterios do 01_build_groups.py:
MD5 identico, dHash-256 com distancia de Hamming <= 24, ou correlacao de miniatura 16x16 >= 0.90.

Uso (na pasta flag-main - v0):  python check_overlap.py
Saidas: results/overlap_check.csv e um resumo no terminal.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAMMING_MAX = 24
CORR_MIN = 0.90


def list_images(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT)


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def features(p):
    im = Image.open(p).convert("L")
    d = np.asarray(im.resize((17, 16), Image.BILINEAR), dtype=np.float32)
    dh = (d[:, 1:] > d[:, :-1]).flatten()  # 16x16 = 256 bits
    t = np.asarray(im.resize((16, 16), Image.BILINEAR), dtype=np.float32).flatten()
    t = t - t.mean()
    n = np.linalg.norm(t)
    t = t / n if n > 0 else t
    return dh, t


def main():
    mar_dir = ROOT / "maritime" / "frames"
    ds_dir = ROOT / "dataset"
    mar = list_images(mar_dir)
    ds = list_images(ds_dir)
    print(f"Fotos maritimas: {len(mar)} | imagens do dataset interno: {len(ds)}")
    if not mar or not ds:
        print("ERRO: pasta vazia. Confira os caminhos maritime/frames e dataset.")
        return

    print("Calculando assinaturas...")
    ds_md5 = {}
    ds_dh, ds_t = [], []
    for p in ds:
        ds_md5.setdefault(md5(p), p)
        a, b = features(p)
        ds_dh.append(a)
        ds_t.append(b)
    ds_dh = np.stack(ds_dh)
    ds_t = np.stack(ds_t)

    rows = []
    for p in mar:
        a, b = features(p)
        ham = (ds_dh != a).sum(axis=1)
        corr = ds_t @ b
        i_h = int(ham.argmin())
        i_c = int(corr.argmax())
        exact = md5(p) in ds_md5
        flag = exact or ham[i_h] <= HAMMING_MAX or corr[i_c] >= CORR_MIN
        rows.append({
            "maritime_photo": p.name,
            "md5_identical": int(exact),
            "min_hamming": int(ham[i_h]),
            "closest_by_hash": str(ds[i_h].relative_to(ds_dir)),
            "max_corr": round(float(corr[i_c]), 4),
            "closest_by_corr": str(ds[i_c].relative_to(ds_dir)),
            "near_duplicate": int(flag),
        })

    out = ROOT / "results" / "overlap_check.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_flag = sum(r["near_duplicate"] for r in rows)
    n_md5 = sum(r["md5_identical"] for r in rows)
    print("\n=== RESULTADO ===")
    print(f"MD5 identico ao dataset interno : {n_md5} de {len(rows)}")
    print(f"Quase-duplicatas (criterio 01)  : {n_flag} de {len(rows)}")
    print(f"Menor distancia de Hamming      : {min(r['min_hamming'] for r in rows)}")
    print(f"Maior correlacao de miniatura   : {max(r['max_corr'] for r in rows)}")
    if n_flag:
        print("\nFotos marcadas:")
        for r in rows:
            if r["near_duplicate"]:
                print(f"  {r['maritime_photo']}  ~  {r['closest_by_hash']} "
                      f"(hamming {r['min_hamming']}, corr {r['max_corr']})")
    print(f"\nDetalhes salvos em: {out}")


if __name__ == "__main__":
    main()
