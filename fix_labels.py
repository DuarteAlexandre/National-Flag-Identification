"""
fix_labels.py  -  corrige os rótulos exportados do CVAT (rode UMA vez, antes do 12_prepare_maritime.py).
1) Faz cópia de segurança de maritime/labels em maritime/labels_backup.
2) Remove linhas DUPLICADAS (cada foto saiu com a mesma caixa 2 vezes).
3) Preenche 3 fotos que ficaram sem caixa (caixas estimadas olhando a imagem).
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT / "maritime" / "labels"
BAK = ROOT / "maritime" / "labels_backup"
if not BAK.exists():
    shutil.copytree(LAB, BAK)
    print("Backup criado em", BAK)

MISSING = {  # classe: Mexico=14, Norway=16, UK=23  (x_centro y_centro largura altura, normalizados)
    "mexico09_000000.txt": "14 0.363000 0.278000 0.108000 0.202000",
    "norway01_000000.txt": "16 0.530000 0.296000 0.859000 0.533000",
    "uk06_000000.txt": "23 0.351000 0.310000 0.320000 0.336000",
}
removed = filled = 0
for f in sorted(LAB.glob("*.txt")):
    lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
    uniq = list(dict.fromkeys(lines))
    removed += len(lines) - len(uniq)
    if not uniq and f.name in MISSING:
        uniq = [MISSING[f.name]]
        filled += 1
    f.write_text("".join(l + "\n" for l in uniq))
print(f"{removed} linhas duplicadas removidas; {filled} fotos preenchidas.")
neg = sum(1 for f in LAB.glob("neg*.txt") if f.read_text().strip() == "")
pos = sum(1 for f in LAB.glob("*.txt") if not f.name.startswith("neg") and f.read_text().strip())
print(f"Resultado: {pos} fotos com bandeira, {neg} fotos sem bandeira (esperado: 267 e 25).")
