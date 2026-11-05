"""
11_import_photos.py
Importa fotos/prints coletados manualmente (pasta "Novos dados/<País>/...") para o pipeline marítimo:
converte .jfif/.webp/.avif/.png/... para JPEG, corrige a rotação EXIF, limita a largura (padrão 1280 px) e
renomeia para  <país><nn>_000000.jpg  (ex.: italy01_000000.jpg)  em  maritime/frames/.

Estrutura esperada:
    Novos dados/
        Italy/ italy1.webp, italy2.jpeg, ...      (nome da pasta = nome EXATO da classe em dataset/)
        Norway/ ...
        _sem_bandeira/ foto1.jpg, ...             (NEGATIVOS: cenas marítimas SEM nenhuma bandeira visível)

Saídas:
    maritime/frames/<id>_000000.jpg
    maritime/photos_manifest.csv      (novo nome <- arquivo original, pasta/país, tamanho original)
    maritime/sources_photos.csv       (OPCIONAL, só com --with-sources: planilha de procedência foto a foto)
    + avisos de fotos duplicadas

Uso:
    python 11_import_photos.py --src "D:/.../Novos dados"
"""
import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".avif", ".gif", ".tif", ".tiff"}
NEG_DIR = "_sem_bandeira"

try:                                    # suporte opcional a AVIF no Pillow antigo (pip install pillow-avif-plugin)
    import pillow_avif  # noqa: F401
except Exception:
    pass


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def dhash(img):
    g = np.asarray(img.convert("L").resize((17, 16), Image.BICUBIC), dtype=np.float32)
    return (g[:, 1:] > g[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help='pasta "Novos dados"')
    ap.add_argument("--out", default=str(ROOT / "maritime" / "frames"))
    ap.add_argument("--width", type=int, default=1280, help="largura máxima (só reduz, nunca amplia)")
    ap.add_argument("--with-sources", action="store_true",
                    help="(opcional) gera maritime/sources_photos.csv para registrar a procedência foto a foto")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted(p.name for p in (ROOT / "dataset").iterdir() if p.is_dir())
    rows, hashes, md5s, skipped = [], [], {}, []
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        if sub.name != NEG_DIR and sub.name not in classes:
            print(f"AVISO: pasta '{sub.name}' não é uma classe de dataset/ nem '{NEG_DIR}' -> ignorada")
            continue
        code = "neg" if sub.name == NEG_DIR else re.sub(r"[^a-z]", "", sub.name.lower())
        files = sorted((f for f in sub.iterdir() if f.suffix.lower() in EXT), key=natural_key)
        for n, f in enumerate(files, 1):
            try:
                im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            except Exception as e:
                skipped.append((str(f), f"{type(e).__name__}: {e}"))
                continue
            w0, h0 = im.size
            if w0 > args.width:
                im = im.resize((args.width, int(round(h0 * args.width / w0))), Image.LANCZOS)
            name = f"{code}{n:02d}_000000.jpg"
            im.save(out / name, quality=93)
            h = dhash(im)
            md = hashlib.md5(f.read_bytes()).hexdigest()
            for (pn, ph) in hashes:
                if int((ph != h).sum()) <= 12:
                    print(f"AVISO: possível FOTO DUPLICADA: {name} ~ {pn}")
            if md in md5s:
                print(f"AVISO: arquivo IDÊNTICO: {name} == {md5s[md]}")
            md5s[md] = name
            hashes.append((name, h))
            rows.append([name, str(f.relative_to(src)), sub.name, w0, h0, im.width, im.height, f.stat().st_size])
        print(f"{sub.name}: {len(files)} arquivos")

    mdir = ROOT / "maritime"
    with open(mdir / "photos_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "original_file", "folder_label", "orig_w", "orig_h", "w", "h", "bytes"])
        w.writerows(rows)
    sp = mdir / "sources_photos.csv"
    if args.with_sources and not sp.exists():   # opcional; nunca sobrescreve o que você já preencheu
        with open(sp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "original_file", "folder_label", "url", "site_or_photographer", "license", "accessed_on", "notes"])
            for r in rows:
                w.writerow([r[0], r[1], r[2], "", "", "", "", ""])
    print(f"\n{len(rows)} imagens importadas para {out}")
    if skipped:
        print(f"{len(skipped)} arquivos NÃO convertidos (provável .avif sem suporte: 'pip install pillow-avif-plugin' ou "
              f"salve como JPG no navegador/Paint):")
        for s in skipped:
            print("   ", s)
    if args.with_sources:
        print(f"Procedência (opcional): preencha {sp}.")


if __name__ == "__main__":
    main()
