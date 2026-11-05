"""
10_extract_frames.py
Extrai frames espaçados de vídeos marítimos NOVOS para montar o conjunto de teste marítimo independente.

Antes de rodar:
  1. Coloque os vídeos em  maritime/videos/  com nomes curtos e estáveis:  v01.mp4, v02.mp4, ...
     (o nome do arquivo vira o "video_id"; o split treino/val/teste será feito POR VÍDEO).
  2. Preencha maritime/sources.csv (video_id, url, canal/fonte, licença, data de acesso, observações).

Uso:
    python 10_extract_frames.py
    python 10_extract_frames.py --every-sec 3 --max-per-video 40 --width 1280

Saída: maritime/frames/<video_id>_<nº do frame>.jpg   e   maritime/frames_manifest.csv
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def dhash(img_bgr):
    g = cv2.cvtColor(cv2.resize(img_bgr, (17, 16), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return (g[:, 1:] > g[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "maritime" / "videos"))
    ap.add_argument("--out", default=str(ROOT / "maritime" / "frames"))
    ap.add_argument("--every-sec", type=float, default=2.0, help="intervalo mínimo entre frames salvos")
    ap.add_argument("--max-per-video", type=int, default=40, help="máx. de frames por vídeo (espaçados uniformemente)")
    ap.add_argument("--width", type=int, default=1280, help="largura final dos frames (mantém proporção)")
    ap.add_argument("--min-diff", type=int, default=20,
                    help="distância de Hamming (dHash 256 bits) mínima entre frames consecutivos salvos (evita quase-repetidos)")
    args = ap.parse_args()

    vdir, odir = Path(args.videos), Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in vdir.iterdir() if p.suffix.lower() in VIDEO_EXT) if vdir.exists() else []
    if not vids:
        raise SystemExit(f"Nenhum vídeo encontrado em {vdir}")

    rows = []
    for v in vids:
        vid = v.stem
        if "_" in vid:
            raise SystemExit(f"Nome de vídeo '{v.name}' contém '_' (reservado). Renomeie para algo como v01.mp4")
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / fps
        step_sec = max(args.every_sec, dur / args.max_per_video)
        step = max(1, int(round(step_sec * fps)))
        saved, last_h, fid = 0, None, 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if fid % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (args.width, int(round(h * args.width / w))), interpolation=cv2.INTER_AREA)
                    hh = dhash(frame)
                    if last_h is None or int((hh != last_h).sum()) >= args.min_diff:
                        name = f"{vid}_{fid:06d}.jpg"
                        cv2.imwrite(str(odir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        rows.append([name, vid, fid, round(fid / fps, 2)])
                        last_h, saved = hh, saved + 1
            fid += 1
        cap.release()
        print(f"{v.name}: {dur:.0f}s, {fps:.1f} fps -> {saved} frames salvos (passo {step_sec:.1f}s)")

    with open(ROOT / "maritime" / "frames_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "video_id", "frame_index", "time_sec"])
        w.writerows(rows)
    print(f"\nTotal: {len(rows)} frames em {odir}")


if __name__ == "__main__":
    main()
