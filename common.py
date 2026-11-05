"""
common.py  -  utilidades compartilhadas pelos scripts da revisão (IMTS, 1a rodada).

Mudanças metodológicas em relação aos scripts originais (resposta aos revisores):
  * NENHUMA augmentation offline: a augmentation é aplicada ONLINE, somente no treino,
    depois do split (corrige o vazamento apontado em R1.2, R3.2).
  * Split agrupado (imagens quase-duplicadas ficam sempre no mesmo subconjunto) e
    estratificado por classe, com semente explícita.
  * Cada subconjunto tem sua própria transformação (o bug do `random_split`, em que
    treino/validação/teste compartilhavam o mesmo `transform`, foi eliminado).
  * O conjunto de teste NUNCA é usado para treinar o YOLO, escolher hiperparâmetros,
    escolher early stopping ou escolher as "classes difíceis" (R1.3).
"""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tvm
from torchvision import transforms as T

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
SPLITS_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"
WORK_DIR = ROOT / "work"
MARITIME_DIR = ROOT / "maritime"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CACHE_SIZE = 256      # lado (px) das imagens em cache (quadradas, como no código original)
IMG_SIZE = 224        # entrada da rede (original: 512; 224 é o padrão ImageNet e ~5x mais rápido em CPU)

# Lista FIXA de classes do dataset interno (ordem alfabética = ordem do ImageFolder original)
# + a classe auxiliar "other" usada apenas na anotação marítima (bandeira de país fora dos 25).
OTHER_NAME = "other"


def get_classes() -> list[str]:
    return sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())


EXCLUDE_FILE = ROOT / "dataset_exclude.txt"   # opcional: um caminho relativo por linha (ex.: dataset/Belgium/4.png)


def list_dataset():
    """Retorna ([(caminho_relativo, id_classe), ...], classes). Ordem determinística.
    Imagens listadas em dataset_exclude.txt (rótulos errados confirmados na auditoria) são ignoradas;
    a pasta dataset/ NUNCA é modificada."""
    classes = get_classes()
    excl = set()
    if EXCLUDE_FILE.exists():
        excl = {l.strip().replace("\\", "/") for l in EXCLUDE_FILE.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")}
    items = []
    for ci, c in enumerate(classes):
        for f in sorted((DATASET_DIR / c).iterdir()):
            rel = str(f.relative_to(ROOT)).replace("\\", "/")
            if f.suffix.lower() in IMG_EXT and rel not in excl:
                items.append((rel, ci))
    if excl:
        print(f"[list_dataset] {len(excl)} imagens excluídas via dataset_exclude.txt")
    return items, classes


# ----------------------------------------------------------------------------
# Utilidades gerais
# ----------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> str:
    # A GPU Intel Iris Xe NÃO é usada pelo PyTorch padrão -> roda em CPU.
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": pick_device(),
    }
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return info


def load_rgb(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_cache_size(img: Image.Image, size: int = CACHE_SIZE) -> np.ndarray:
    return np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def center_crop_pil(img: Image.Image, ratio: float) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def crop_box(img: Image.Image, xyxy, margin: float = 0.0) -> Image.Image | None:
    """Recorta [x1,y1,x2,y2] (pixels) com margem relativa; devolve None se vazio."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw, bh = x2 - x1, y2 - y1
    x1 -= margin * bw
    x2 += margin * bw
    y1 -= margin * bh
    y2 += margin * bh
    W, H = img.size
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(W, int(np.ceil(x2))), min(H, int(np.ceil(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img.crop((x1, y1, x2, y2))


def build_cache(paths, fn, name: str, force: bool = False) -> np.ndarray:
    """Pré-processa todas as imagens com `fn(PIL)->PIL` e guarda (N,S,S,3) uint8 em disco."""
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / f"{name}.npy"
    if f.exists() and not force:
        arr = np.load(f)
        if len(arr) == len(paths):
            return arr
    t0 = time.time()
    out = np.zeros((len(paths), CACHE_SIZE, CACHE_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        out[i] = to_cache_size(fn(load_rgb(ROOT / p)))
        if (i + 1) % 200 == 0:
            print(f"   cache '{name}': {i + 1}/{len(paths)}")
    np.save(f, out)
    print(f"   cache '{name}' pronto ({time.time() - t0:.0f}s)")
    return out


# ----------------------------------------------------------------------------
# Split agrupado + estratificado
# ----------------------------------------------------------------------------
def group_split(labels, groups, seed: int, n_test: int = 7, n_val: int = 6):
    """
    ~71% treino / ~14% validação / ~14% teste, estratificado por classe e AGRUPADO:
    todas as imagens de um mesmo grupo (quase-duplicatas) caem no mesmo subconjunto.
    Devolve (train_idx, val_idx, test_idx).
    """
    from sklearn.model_selection import StratifiedGroupKFold
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    idx = np.arange(len(labels))
    s1 = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    rest, test = next(iter(s1.split(idx, labels, groups)))
    s2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed + 1000)
    tr, va = next(iter(s2.split(rest, labels[rest], groups[rest])))
    train, val = rest[tr], rest[va]
    for a, b in ((train, val), (train, test), (val, test)):
        assert not (set(groups[a]) & set(groups[b])), "vazamento de grupo entre subconjuntos!"
    return np.sort(train), np.sort(val), np.sort(test)


# ----------------------------------------------------------------------------
# Transformações / Dataset
# ----------------------------------------------------------------------------
def make_transforms(img_size: int = IMG_SIZE) -> dict:
    norm = T.Normalize(MEAN, STD)
    return {
        "eval": T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm]),
        # política "moderada": usada em 'uniform' (todas as classes) e nas classes não-fracas em 'targeted'
        "moderate": T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2),
            T.ToTensor(), norm]),
        # política "forte": classes fracas em 'targeted'
        "strong": T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(30),
            T.ColorJitter(0.5, 0.5, 0.5, 0.1),
            T.RandomApply([T.GaussianBlur(7, (0.1, 2.0))], p=0.3),
            T.ToTensor(), norm]),
    }


class ArrayDataset(Dataset):
    """
    images: array (N,S,S,3) uint8.  aug in {'none','uniform','targeted'} (só vale com train=True).
    'targeted': classes em `weak` recebem política forte E são repetidas `repeat` vezes por época
                (cada repetição com uma augmentation diferente); as demais, política moderada.
    """

    def __init__(self, images, labels, aug="none", weak=(), repeat=3, train=False, img_size=IMG_SIZE):
        self.images = images
        self.labels = np.asarray(labels).astype(int)
        self.train = train
        self.aug = aug if train else "none"
        self.weak = set(int(w) for w in weak)
        self.tf = make_transforms(img_size)
        idx = []
        for i, y in enumerate(self.labels):
            n = repeat if (self.train and self.aug == "targeted" and int(y) in self.weak) else 1
            idx.extend([i] * n)
        self.index = idx

    def extra_samples_per_epoch(self) -> int:
        return len(self.index) - len(self.labels)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = self.index[k]
        y = int(self.labels[i])
        img = Image.fromarray(self.images[i])
        if self.aug == "none":
            tf = self.tf["eval"]
        elif self.aug == "uniform":
            tf = self.tf["moderate"]
        else:
            tf = self.tf["strong"] if y in self.weak else self.tf["moderate"]
        return tf(img), y


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
ARCHS = ["resnet18", "resnet50", "mobilenet_v3_large", "efficientnet_b0"]


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Mesma filosofia do código original: backbone congelado + último bloco + cabeça treináveis."""
    if arch == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        trainable = [m.layer4]
    elif arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        trainable = [m.layer4]
    elif arch == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        trainable = [m.features[-3:], m.classifier]
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        trainable = [m.features[-2:]]
    else:
        raise ValueError(arch)
    for p in m.parameters():
        p.requires_grad = False
    for mod in trainable:
        for p in mod.parameters():
            p.requires_grad = True
    # nova cabeça (treinável por padrão)
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def _set_train(model: nn.Module):
    """model.train(), mas BatchNorm de camadas congeladas permanece em eval (não atualiza estatísticas)."""
    model.train()
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in mod.parameters()):
                mod.eval()


@torch.no_grad()
def predict_probs(model, dataset, device, batch=64) -> np.ndarray:
    """Probabilidades softmax (N,C) na ordem do dataset (use datasets de avaliação, sem repetição)."""
    model.eval().to(device)
    dl = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    out = []
    for x, _ in dl:
        out.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def train_classifier(model, train_ds, val_ds, device, epochs=30, patience=7, lr=3e-4, batch=32, log=print):
    """Adam + early stopping na PERDA DE VALIDAÇÃO. Devolve (modelo com melhores pesos, histórico)."""
    model.to(device)
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()
    best = {"val_loss": float("inf"), "val_acc": 0.0, "epoch": 0}
    best_state, bad, hist = None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        _set_train(model)
        tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        model.eval()
        vloss, vcorrect, vn = 0.0, 0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vloss += crit(out, y).item() * len(y)
                vcorrect += (out.argmax(1) == y).sum().item()
                vn += len(y)
        vloss, vacc = vloss / max(vn, 1), vcorrect / max(vn, 1)
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": vloss, "val_acc": vacc})
        if vloss < best["val_loss"] - 1e-4:
            best = {"val_loss": vloss, "val_acc": vacc, "epoch": ep}
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        log(f"      ep {ep:02d} | train {tot / max(n, 1):.4f} | val_loss {vloss:.4f} | val_acc {vacc:.3f}"
            f"{'  *' if bad == 0 else ''}  ({time.time() - t0:.0f}s)")
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best["epoch"], "epochs_run": len(hist), "best_val_loss": best["val_loss"],
                   "best_val_acc": best["val_acc"], "train_seconds": time.time() - t0, "history": hist}


def classification_metrics(y_true, y_pred, classes, present_only=False) -> dict:
    """present_only=True: médias macro apenas sobre as classes presentes em y_true (teste marítimo)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    all_labels = list(range(len(classes)))
    labels = sorted(set(y_true.tolist())) if present_only else all_labels
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    pc, rc, fc, sup = precision_recall_fscore_support(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f), "f1_weighted": float(fw),
        "per_class": {classes[i]: {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]),
                                   "support": int(sup[i])} for i in all_labels},
        "confusion_matrix": cm.tolist(),
    }


def time_calls(fn, inputs, warmup=3):
    """Tempo médio (ms) e desvio-padrão de fn(x) para cada x; descarta `warmup` primeiras chamadas."""
    for x in inputs[:warmup]:
        fn(x)
    ts = []
    for x in inputs:
        t0 = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": float(np.mean(ts)), "std_ms": float(np.std(ts)), "n": len(ts)}


@torch.no_grad()
def classify_pil(model, pil_img, device, img_size=IMG_SIZE, tf=None):
    """Classifica UMA imagem PIL (lote=1). Devolve vetor de probabilidades (numpy)."""
    tf = tf or make_transforms(img_size)["eval"]
    x = tf(pil_img).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=1)[0].cpu().numpy()


@torch.no_grad()
def classify_pil_batch(model, pil_imgs, device, img_size=IMG_SIZE):
    tf = make_transforms(img_size)["eval"]
    x = torch.stack([tf(im) for im in pil_imgs]).to(device)
    return torch.softmax(model(x), dim=1).cpu().numpy()


# ----------------------------------------------------------------------------
# YOLO (ultralytics)
# ----------------------------------------------------------------------------
def write_yolo_yaml(yaml_path: Path, root: Path, names=("flag",), train="images/train", val="images/val", test=None):
    lines = [f"path: {root.resolve().as_posix()}", f"train: {train}", f"val: {val}"]
    if test:
        lines.append(f"test: {test}")
    lines.append("names:")
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_yolo(data_yaml, project, name, weights, epochs, imgsz, batch, seed, device, workers=0, patience=20):
    from ultralytics import YOLO
    model = YOLO(weights)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, seed=seed, device=device,
                workers=workers, project=str(Path(project).resolve()), name=name, exist_ok=True,
                plots=False, verbose=False, patience=patience)
    return Path(model.trainer.best)


def yolo_val(best_pt, data_yaml, split, imgsz, device, project, name) -> dict:
    """mAP explícito: map50 = mAP@0.5 ; map50_95 = mAP@0.5:0.95."""
    from ultralytics import YOLO
    m = YOLO(str(best_pt))
    r = m.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device, plots=False, verbose=False,
              project=str(Path(project).resolve()), name=name, exist_ok=True)
    return {"split": split, "map50": float(r.box.map50), "map50_95": float(r.box.map),
            "precision": float(r.box.mp), "recall": float(r.box.mr)}


def yolo_detect(model, pil_img, imgsz, conf, device):
    """Devolve (boxes_xyxy (n,4) em pixels da imagem original, confs (n,)) já com NMS."""
    r = model.predict(source=pil_img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
