"""Optional: packs ONLY non-image metadata (splits, groups, box labels, attributes, result tables) into
flag_metadata_no_images.zip, so it can be published without any photograph.
Excluded on purpose: dataset/, maritime/frames, qualitative figures, *.jpg/*.jpeg/*.webp, model weights (*.pt).
Usage:  python make_metadata_zip.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OK_EXT = {".csv", ".json", ".md", ".txt"}
items = []
for sub in ["splits", "results"]:
    for f in (ROOT / sub).rglob("*") if (ROOT / sub).exists() else []:
        if f.is_file() and f.suffix.lower() in OK_EXT and "qualitative" not in f.parts and "work" not in f.parts:
            items.append(f)
        if f.is_file() and f.suffix.lower() == ".png" and "qualitative" not in f.parts and "confusion" in f.name:
            items.append(f)
mar = ROOT / "maritime"
for name in ["split.csv", "attributes.csv", "classes.txt", "stats.md"]:
    if (mar / name).exists():
        items.append(mar / name)
if (mar / "labels").exists():
    items += [f for f in (mar / "labels").glob("*.txt")]
out = ROOT / "flag_metadata_no_images.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in items:
        z.write(f, f.relative_to(ROOT).as_posix())
print(f"{len(items)} files -> {out}")
