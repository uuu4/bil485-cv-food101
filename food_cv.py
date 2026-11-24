"""BIL485 Ödev 2: Food-101 ile yemek fotoğrafı sınıflandırma.

Notebook (`food101_cnn.ipynb`) ve Gradio uygulaması (`app.py`) bu modülü ortak kullanır:
veri bölme, modeller, eğitim döngüsü, metrikler, Grad-CAM, telefon fotoğrafı okuma,
checkpoint ve rapor üretimi burada.
"""
from __future__ import annotations

import copy
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:  # iPhone fotoğrafları HEIC gelebilir
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_OK = True
except ImportError:  # pragma: no cover
    HEIC_OK = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}


# --------------------------------------------------------------------------- genel
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretty(name: str) -> str:
    """'french_fries' -> 'French fries'"""
    s = name.replace("_", " ")
    return s[:1].upper() + s[1:]


def environment_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover
        pass
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- görüntü okuma
def load_image(src) -> Image.Image:
    """Dosya yolu ya da PIL görüntüsü -> EXIF yönü düzeltilmiş RGB görüntü.

    Telefon fotoğrafları çoğu zaman yan kaydedilir ve yönü EXIF etiketinde durur;
    düzeltilmezse model yemeği 90 derece dönmüş görür.
    """
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def train_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_view(img_size: int) -> transforms.Compose:
    """Değerlendirmede modelin gördüğü kare (normalizasyondan önce, gösterim için)."""
    return transforms.Compose([
        transforms.Resize(int(round(img_size / 0.875))),
        transforms.CenterCrop(img_size),
    ])


def eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        eval_view(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- veri
class ImageListDataset(Dataset):
    """(dosya yolu, etiket) listesi üzerinde basit veri kümesi."""

    def __init__(self, files, labels, transform):
        assert len(files) == len(labels)
        self.files = list(files)
        self.labels = [int(x) for x in labels]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def stratified_split(labels, val_per_class: int, seed: int):
    """Her sınıftan `val_per_class` örneği validation'a ayırır; geri kalanı train."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        val_idx.extend(idx[:val_per_class].tolist())
        train_idx.extend(idx[val_per_class:].tolist())
    return sorted(train_idx), sorted(val_idx)


def subsample_per_class(indices, labels, per_class: int, seed: int):
    labels = np.asarray(labels)
    indices = np.asarray(indices)
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(labels[indices]):
        idx = indices[labels[indices] == c]
        rng.shuffle(idx)
        out.extend(idx[:per_class].tolist())
    return sorted(out)


def food101_file_lists(root: str | Path, download: bool = True):
    """torchvision Food101'i indirir; (train_files, train_labels, test_files, test_labels, classes) döner.

    Resmî bölünme: sınıf başına 750 train + 250 test görüntüsü (toplam 101 000).
    """
    from torchvision.datasets import Food101

    tr = Food101(str(root), split="train", download=download)
    te = Food101(str(root), split="test", download=download)
    return (
        [str(p) for p in tr._image_files], list(tr._labels),
        [str(p) for p in te._image_files], list(te._labels),
        list(tr.classes),
    )


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False):
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), drop_last=drop_last)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw)


# --------------------------------------------------------------------------- modeller
class SmallCNN(nn.Module):
    """Sıfırdan eğitilen temel CNN: 4 blok (Conv-BN-ReLU x2 + MaxPool), GAP, Dropout, Linear."""

    def __init__(self, num_classes: int, widths=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 3
        for w in widths:
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            c_in = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


def build_resnet50(num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    return m


def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if arch == "resnet50":
        return build_resnet50(num_classes, pretrained=pretrained, dropout=dropout)
    if arch == "smallcnn":
        return SmallCNN(num_classes, dropout=dropout)
    raise ValueError(f"bilinmeyen mimari: {arch}")


def head_parameters(model: nn.Module):
    return list(model.fc.parameters())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def target_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM için son evrişim bloğu."""
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Grad-CAM katmanı bulunamadı")


# --------------------------------------------------------------------------- eğitim
@dataclass
class TrainConfig:
    name: str
    arch: str                       # "resnet50" | "smallcnn"
    img_size: int = 224
    epochs: int = 6
    batch_size: int = 64
    lr: float = 1e-3                # sınıflandırıcı başı (head) öğrenme oranı
    backbone_lr_mult: float = 0.1   # gövde lr = lr * bu çarpan (sıfırdan eğitimde 1.0)
    weight_decay: float = 1e-4
    warmup_epochs: float = 0.5
    label_smoothing: float = 0.1
    optimizer: str = "AdamW"
    pretrained: bool = True
    dropout: float = 0.2
    amp: bool = True


def make_optimizer(model: nn.Module, cfg: TrainConfig):
    head = head_parameters(model)
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    groups = [
        {"params": body, "lr": cfg.lr * cfg.backbone_lr_mult},
        {"params": head, "lr": cfg.lr},
    ]
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def warmup_cosine(total_steps: int, warmup_steps: int):
    """Adım başına lr çarpanı: doğrusal ısınma, ardından kosinüsle sıfıra iniş."""
    def f(step):
        if warmup_steps > 0 and step < warmup_steps:
            return 0.01 + 0.99 * step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, max_batches=None, return_outputs=False, amp=True):
    model.eval()
    total, correct, correct5, loss_sum = 0, 0, 0, 0.0
    all_logits, all_labels = [], []
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with _autocast(device, amp):
            logits = model(x)
        logits = logits.float()
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.size(0)
        k = min(5, logits.size(1))
        top = logits.topk(k, dim=1).indices
        correct += (top[:, 0] == y).sum().item()
        correct5 += (top == y[:, None]).any(1).sum().item()
        total += y.size(0)
        if return_outputs:
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    out = {"loss": loss_sum / max(1, total) if criterion is not None else None,
           "acc": correct / max(1, total), "top5": correct5 / max(1, total), "n": total}
    if return_outputs:
        out["logits"] = torch.cat(all_logits).numpy()
        out["labels"] = torch.cat(all_labels).numpy()
    return out


def fit(model, train_loader, val_loader, cfg: TrainConfig, device, max_train_batches=None,
        max_eval_batches=None, log_every: int = 100):
    """Modeli eğitir; validation doğruluğu en yüksek epoch'un ağırlıklarını geri yükler.

    Döner: (history listesi, en iyi epoch, toplam eğitim süresi sn)
    """
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    opt = make_optimizer(model, cfg)
    steps_per_epoch = len(train_loader) if max_train_batches is None else min(len(train_loader), max_train_batches)
    total_steps = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total_steps, int(cfg.warmup_epochs * steps_per_epoch)))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_acc, best_epoch, best_state = [], -1.0, -1, None
    t_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        seen, correct, loss_sum = 0, 0, 0.0
        for step, (x, y) in enumerate(train_loader):
            if max_train_batches is not None and step >= max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with _autocast(device, use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
            if log_every and (step + 1) % log_every == 0:
                ips = seen / (time.perf_counter() - t0)
                print(f"  [{cfg.name}] epoch {epoch} adım {step + 1}/{steps_per_epoch} "
                      f"loss {loss_sum / seen:.3f} acc {correct / seen:.3f} ({ips:.0f} görüntü/sn)")
        train_time = time.perf_counter() - t0
        val = evaluate(model, val_loader, device, eval_criterion, max_batches=max_eval_batches, amp=cfg.amp)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, seen),
            "train_acc": correct / max(1, seen),
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_top5": val["top5"],
            "lr_head": opt.param_groups[1]["lr"],
            "epoch_time_s": time.perf_counter() - t0,
            "train_time_s": train_time,
        }
        history.append(row)
        print(f"[{cfg.name}] epoch {epoch}/{cfg.epochs}  train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}"
              f"  |  val loss {row['val_loss']:.3f} acc {row['val_acc']:.4f} top5 {row['val_top5']:.4f}"
              f"  |  {row['epoch_time_s'] / 60:.1f} dk")
        if val["acc"] > best_acc:
            best_acc, best_epoch = val["acc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    total_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{cfg.name}] bitti: en iyi epoch {best_epoch} (val acc {best_acc:.4f}), toplam {total_time / 60:.1f} dk")
    return history, best_epoch, total_time


# --------------------------------------------------------------------------- metrikler
def classification_metrics(logits: np.ndarray, labels: np.ndarray, classes: list[str], n_pairs: int = 10) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    C = len(classes)
    preds = logits.argmax(1)
    k = min(5, C)
    top5 = (np.argsort(-logits, axis=1)[:, :k] == labels[:, None]).any(1).mean()
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    pc_p, pc_r, pc_f, pc_s = precision_recall_fscore_support(labels, preds, labels=list(range(C)), average=None,
                                                             zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(C)))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:n_pairs]
    pairs = []
    for idx in flat:
        i, j = np.unravel_index(idx, off.shape)
        if off[i, j] == 0:
            break
        pairs.append({"true": classes[i], "pred": classes[j], "count": int(off[i, j]),
                      "rate": float(off[i, j] / max(1, cm[i].sum()))})
    per_class = [{"class": classes[c], "precision": float(pc_p[c]), "recall": float(pc_r[c]),
                  "f1": float(pc_f[c]), "support": int(pc_s[c])} for c in range(C)]
    return {
        "n": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "top5_accuracy": float(top5),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "per_class": per_class,
        "top_confused_pairs": pairs,
        "confusion_matrix": cm,
    }


def time_inference(model, device, img_size: int, n_warmup: int = 10, n_runs: int = 50, batch_size: int = 1) -> float:
    """Görüntü başına medyan çıkarım süresi (ms). Ön işleme dahil değil."""
    model = model.eval().to(device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    times = []
    with torch.inference_mode():
        for i in range(n_warmup + n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= n_warmup:
                times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000 / batch_size)


# --------------------------------------------------------------------------- checkpoint
def save_checkpoint(model, cfg: TrainConfig, classes: list[str], path, extra: dict | None = None) -> Path:
    """Ağırlıkları fp16 saklar (ResNet-50 ~47 MB, GitHub'ın 100 MB sınırının altında)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
             for k, v in model.state_dict().items()}
    torch.save({"config": asdict(cfg), "classes": list(classes), "state_dict": state,
                "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "extra": extra or {}}, path)
    return path


def load_checkpoint(path, device: torch.device | str = "cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg.arch, len(ckpt["classes"]), pretrained=False, dropout=cfg.dropout)
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["state_dict"].items()})
    model.eval().to(device)
    return model, {"config": cfg, "classes": ckpt["classes"], "extra": ckpt.get("extra", {})}


# --------------------------------------------------------------------------- tek görüntü tahmini + Grad-CAM
@torch.no_grad()
def predict_image(model, img: Image.Image, classes, img_size: int, device, topk: int = 5):
    model.eval()
    x = eval_transform(img_size)(img).unsqueeze(0).to(device)
    probs = model(x).float().softmax(1)[0].cpu()
    p, idx = probs.topk(min(topk, len(classes)))
    return [(classes[i], float(q)) for q, i in zip(p.tolist(), idx.tolist())]


def grad_cam(model, img: Image.Image, img_size: int, device, class_idx: int | None = None):
    """Grad-CAM ısı haritası: son evrişim bloğunun aktivasyonları, hedef sınıfın gradyanıyla ağırlıklanır.

    Döner: (img_size x img_size, 0-1 aralığında harita, modelin gördüğü kare PIL görüntü, sınıf indeksi)
    """
    model.eval()
    view = eval_view(img_size)(img)
    x = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(view)
    x = x.unsqueeze(0).to(device)
    store = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = target_layer(model).register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            x.requires_grad_(True)
            logits = model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1))
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
    finally:
        h.remove()
    act, grad = store["act"].detach(), store["grad"].detach()
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(img_size, img_size), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    model.zero_grad(set_to_none=True)
    return cam.cpu().numpy(), view, class_idx


def overlay_cam(view: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib

    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(view.convert("RGB")).astype(np.float32)
    mix = (1 - alpha) * base + alpha * heat.astype(np.float32)
    return Image.fromarray(mix.clip(0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- kendi fotoğraflarım
def list_images(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def convert_heic_to_jpeg(folder) -> list[Path]:
    """HEIC/HEIF fotoğrafları yönü düzeltilmiş JPEG'e çevirir (tarayıcılar HEIC'i çoğunlukla gösteremez)."""
    out = []
    for p in list_images(folder):
        if p.suffix.lower() in {".heic", ".heif"}:
            dst = p.with_suffix(".jpg")
            load_image(p).save(dst, quality=95)
            p.unlink()
            out.append(dst)
    return out


def label_from_filename(path) -> str:
    """'pizza.jpg' -> 'pizza', 'french_fries-2.HEIC' -> 'french_fries', 'Mercimek Corbasi.jpg' -> 'mercimek_corbasi'"""
    stem = Path(path).stem.split("-")[0].strip().lower()
    return "_".join(stem.split())


def evaluate_my_photos(model, folder, classes, img_size: int, device, topk: int = 5) -> list[dict]:
    rows = []
    for p in list_images(folder):
        img = load_image(p)
        preds = predict_image(model, img, classes, img_size, device, topk=topk)
        true = label_from_filename(p)
        in_classes = true in classes
        rows.append({
            "file": p.name,
            "true_label": true,
            "in_food101": in_classes,
            "pred_top1": preds[0][0],
            "pred_top1_prob": preds[0][1],
            "top5": preds,
            "correct_top1": in_classes and preds[0][0] == true,
            "correct_top5": in_classes and any(c == true for c, _ in preds),
            "true_rank": next((i + 1 for i, (c, _) in enumerate(preds) if c == true), None),
            "original_size": img.size,
        })
    return rows


# --------------------------------------------------------------------------- rapor
def fmt_pct(x) -> str:
    return "-" if x is None else f"%{100 * x:.2f}".replace(".", ",")


def tr(x, fmt: str = "g") -> str:
    """Türkçe ondalık virgül: tr(23.5, '.1f') -> '23,5'"""
    return format(x, fmt).replace(".", ",")


def report_table(r: dict) -> list[tuple[str, str]]:
    """Ödev belgesindeki 9. bölüm şablonunun 16 satırı, results.json'dan doldurulur."""
    best = r["selected_model"]
    cfg = r["models"][best]["config"]
    test = r["models"][best]["test"]
    env = r["environment"]
    hp = (f"epoch {cfg['epochs']}, batch {cfg['batch_size']}, lr (head) {tr(cfg['lr'])}, "
          f"lr (gövde) {tr(cfg['lr'] * cfg['backbone_lr_mult'])}, weight decay {tr(cfg['weight_decay'])}, "
          f"warmup {tr(cfg['warmup_epochs'])} epoch + kosinüs azalma, label smoothing {tr(cfg['label_smoothing'])}, "
          f"dropout {tr(cfg['dropout'])}, seed {r['seed']}, karma hassasiyet (AMP fp16)")
    n_classes = len(r.get("classes") or []) or 101
    train_min = r["models"][best]["train_time_s"] / 60
    inf = r["models"][best]["inference_ms"]
    return [
        ("Project Title", r["project_title"]),
        ("Problem / Task", "Görüntü sınıflandırma: bir yemek fotoğrafını 101 yemek sınıfından birine atama"),
        ("Dataset", f"Food-101 (101 sınıf, 101 000 görüntü). Bölme: train {r['split']['train']}, "
                    f"validation {r['split']['val']} (resmî train'den sınıf başına {r['split']['val_per_class']}, "
                    f"tabakalı), test {r['split']['test']} (resmî test, yalnız sonda bir kez)"),
        ("Dataset URL", "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/"),
        ("Model Architecture", f"{r['models'][best]['architecture']} ({tr(r['models'][best]['params_m'], '.1f')} M parametre); "
                               f"karşılaştırma: {', '.join(m['architecture'] for k, m in r['models'].items() if k != best)}"),
        ("Input Size", f"3 x {cfg['img_size']} x {cfg['img_size']} (RGB, ImageNet ortalama/std ile normalize)"),
        ("Hyperparameters", hp),
        ("Optimizer", f"{cfg['optimizer']} (gövde ve sınıflandırıcı başı için ayrı öğrenme oranı)"),
        ("Activation / Output Function", f"Gizli katmanlar ReLU; çıkış {n_classes} nöronlu Linear + Softmax"),
        ("Loss Function", f"Cross-entropy (label smoothing {tr(cfg['label_smoothing'])})"),
        ("Evaluation Metrics", "Accuracy, top-5 accuracy, macro precision / recall / F1, karışıklık matrisi"),
        ("Training Environment", f"Google Colab, {env['device']}, PyTorch {env['torch']}, "
                                 f"eğitim süresi {tr(train_min, '.1f')} dk"),
        ("Best Performance", f"Test accuracy {fmt_pct(test['accuracy'])}, top-5 {fmt_pct(test['top5_accuracy'])}, "
                             f"macro F1 {fmt_pct(test['macro_f1'])}"),
        ("Inference Time", f"Görüntü başına {tr(inf['gpu_ms'], '.1f')} ms (GPU, batch 1), {inf['cpu_ms']:.0f} ms (CPU, batch 1)"
         if inf.get("gpu_ms") is not None else f"Görüntü başına {inf['cpu_ms']:.0f} ms (CPU, batch 1)"),
        ("Deployment Technology", "Gradio web uygulaması (`app.py`): fotoğraf yükleme/kamera, ilk 5 tahmin, Grad-CAM"),
        ("Repository", r["repository"]),
    ]


def write_report(r: dict, path) -> Path:
    lines = ["# Proje Raporu: " + r["project_title"], "",
             "Bu tablo `food101_cnn.ipynb` tarafından `results/results.json`'dan otomatik üretilir.", "",
             "| Alan | Açıklama / Öğrenci yanıtı |", "|---|---|"]
    for k, v in report_table(r):
        lines.append(f"| {k} | {str(v).replace('|', '/')} |")
    lines += ["", "## Model karşılaştırması (aynı train/validation/test bölmesi)", "",
              "| Model | Parametre | En iyi epoch | Val acc | Test acc | Test top-5 | Test macro F1 | Eğitim süresi |",
              "|---|---|---|---|---|---|---|---|"]
    for k, m in r["models"].items():
        t = m["test"]
        lines.append(f"| {m['architecture']} | {tr(m['params_m'], '.1f')} M | {m['best_epoch']} | {fmt_pct(m['best_val_acc'])} | "
                     f"{fmt_pct(t['accuracy'])} | {fmt_pct(t['top5_accuracy'])} | {fmt_pct(t['macro_f1'])} | "
                     f"{tr(m['train_time_s'] / 60, '.1f')} dk |")
    photos = r.get("my_photos") or []
    if photos:
        lines += ["", "## Gerçek dünya fotoğrafları (Food-101 dışından)", "",
                  "| Dosya | Gerçek | Food-101'de var mı | Tahmin (olasılık) | Doğru mu | Gerçek sınıfın sırası |",
                  "|---|---|---|---|---|---|"]
        for p in photos:
            ok = "evet" if p["correct_top1"] else ("sınıf dışı" if not p["in_food101"] else "hayır")
            lines.append(f"| {p['file']} | {pretty(p['true_label'])} | {'evet' if p['in_food101'] else 'hayır'} | "
                         f"{pretty(p['pred_top1'])} (%{tr(100 * p['pred_top1_prob'], '.1f')}) | {ok} | "
                         f"{p['true_rank'] if p['true_rank'] else '-'} |")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
