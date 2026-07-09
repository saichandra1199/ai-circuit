import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

import yaml


# ── transforms ────────────────────────────────────────────────────────────────

def build_transforms(cfg):
    aug = cfg.get("augmentations", {})
    sz = cfg["model"]["image_size"]
    resize_sz = int(sz * 256 / 224)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_t = [transforms.Resize(resize_sz)]

    if aug.get("random_resized_crop", True):
        train_t.append(transforms.RandomResizedCrop(sz))
    else:
        train_t.append(transforms.CenterCrop(sz))

    if aug.get("random_horizontal_flip", True):
        train_t.append(transforms.RandomHorizontalFlip(0.5))

    if aug.get("autoaugment", False):
        train_t.append(transforms.AutoAugment())
    elif aug.get("randaugment", False):
        train_t.append(transforms.RandAugment(
            num_ops=aug.get("randaugment_n", 2),
            magnitude=aug.get("randaugment_m", 9),
        ))
    elif aug.get("color_jitter", True):
        train_t.append(transforms.ColorJitter(
            brightness=aug.get("color_jitter_brightness", 0.2),
            contrast=aug.get("color_jitter_contrast", 0.2),
            saturation=aug.get("color_jitter_saturation", 0.2),
            hue=aug.get("color_jitter_hue", 0.05),
        ))

    train_t += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if aug.get("random_erasing", False):
        train_t.append(transforms.RandomErasing(p=aug.get("random_erasing_prob", 0.5)))

    val_t = [
        transforms.Resize(resize_sz),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    return transforms.Compose(train_t), transforms.Compose(val_t)


# ── mixup / cutmix ────────────────────────────────────────────────────────────

def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    ch = int(H * (1 - lam) ** 0.5)
    cw = int(W * (1 - lam) ** 0.5)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
    y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y[idx], lam


# ── loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


def get_criterion(cfg, class_weights_tensor, device):
    loss_cfg = cfg.get("loss", {})
    w = class_weights_tensor.to(device) if loss_cfg.get("use_class_weights", True) else None
    loss_type = loss_cfg.get("type", "weighted_ce")
    if loss_type == "focal":
        return FocalLoss(w, gamma=loss_cfg.get("focal_gamma", 2.0))
    return nn.CrossEntropyLoss(weight=w, label_smoothing=loss_cfg.get("label_smoothing", 0.0))


# ── optimizer + scheduler ─────────────────────────────────────────────────────

def get_optimizer(cfg, model):
    c = cfg["optimizer"]
    t = c.get("type", "adamw").lower()
    p = model.parameters()
    if t == "sgd":
        return torch.optim.SGD(p, lr=c["lr"], weight_decay=c.get("weight_decay", 0.01),
                               momentum=c.get("momentum", 0.9))
    if t == "adam":
        return torch.optim.Adam(p, lr=c["lr"], weight_decay=c.get("weight_decay", 0.01))
    return torch.optim.AdamW(p, lr=c["lr"], weight_decay=c.get("weight_decay", 0.01))


def get_scheduler(cfg, optimizer, steps_per_epoch):
    c = cfg.get("scheduler", {})
    t = c.get("type", "cosine").lower()
    epochs = cfg["training"]["epochs"]
    if t == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=c.get("min_lr", 1e-6))
    if t == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=c.get("step_size", 7), gamma=c.get("gamma", 0.1))
    if t == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["optimizer"]["lr"],
            epochs=epochs, steps_per_epoch=steps_per_epoch)
    if t == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=c.get("patience", 3), factor=c.get("gamma", 0.1))
    return None


# ── train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device, cfg, use_amp, scheduler=None):
    model.train()
    aug = cfg.get("augmentations", {})
    use_mixup = aug.get("mixup", False)
    use_cutmix = aug.get("cutmix", False)
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        ya, yb, lam = None, None, 1.0

        if use_cutmix and use_mixup:
            fn = cutmix if np.random.rand() < 0.5 else mixup
            imgs, ya, yb, lam = fn(imgs, labels, aug.get("cutmix_alpha" if fn is cutmix else "mixup_alpha", 1.0))
        elif use_cutmix:
            imgs, ya, yb, lam = cutmix(imgs, labels, aug.get("cutmix_alpha", 1.0))
        elif use_mixup:
            imgs, ya, yb, lam = mixup(imgs, labels, aug.get("mixup_alpha", 0.4))

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs)
            loss = (lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                    if ya is not None else criterion(logits, labels))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names, use_amp):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs)
            total_loss += criterion(logits, labels).item() * imgs.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    labels = list(range(len(class_names)))
    report = classification_report(all_labels, all_preds, target_names=class_names,
                                   labels=labels, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {c: report[c] for c in class_names},
        "confusion_matrix": cm,
    }


# ── dataset helpers ───────────────────────────────────────────────────────────

def _sample_labels(ds):
    if hasattr(ds, "samples"):
        return [lbl for _, lbl in ds.samples]
    return [ds.dataset.samples[i][1] for i in ds.indices]


def _subsample(ds, max_per_class: int):
    buckets = defaultdict(list)
    for idx, lbl in enumerate(_sample_labels(ds)):
        buckets[lbl].append(idx)
    indices = []
    for lbl in sorted(buckets):
        indices.extend(buckets[lbl][:max_per_class])
    return torch.utils.data.Subset(ds, indices)


# ── main ──────────────────────────────────────────────────────────────────────

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    with open(cfg["paths"]["class_weights"]) as f:
        cw_dict = json.load(f)
    with open(cfg["paths"]["class_mapping"]) as f:
        class_mapping = json.load(f)

    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    cw_tensor = torch.tensor([cw_dict[c] for c in class_names], dtype=torch.float)

    # experiment dir — agent sets paths.output_dir directly; humans get auto-timestamped path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    out_dir = cfg["paths"].get("output_dir")
    if out_dir:
        exp_dir = Path(out_dir)
    else:
        exp_dir = Path(cfg["paths"].get("experiment_dir", "experiments")) / f"{exp_name}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    dst = exp_dir / "config.yaml"
    if Path(config_path).resolve() != dst.resolve():
        shutil.copy(config_path, dst)

    train_tf, val_tf = build_transforms(cfg)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=val_tf)

    tcfg = cfg["training"]
    max_per_class = tcfg.get("max_samples_per_class")
    if max_per_class:
        train_ds = _subsample(train_ds, max_per_class)
        val_ds   = _subsample(val_ds,   max_per_class)
        test_ds  = _subsample(test_ds,  max_per_class)

    train_n = len(train_ds)
    val_n   = len(val_ds)
    print(f"\n[{exp_name}] {cfg['model']['backbone']}  train={train_n}  val={val_n}  epochs={tcfg['epochs']}\n")

    if cfg.get("sampler", {}).get("use_weighted", True):
        labels = _sample_labels(train_ds)
        sw = torch.tensor([cw_dict[class_names[lbl]] for lbl in labels], dtype=torch.float)
        sampler, shuffle = WeightedRandomSampler(sw, len(sw), replacement=True), False
    else:
        sampler, shuffle = None, True

    nw = tcfg.get("num_workers", 0)
    bs = tcfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, shuffle=shuffle,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = tcfg.get("mixed_precision", True) and device.type == "cuda"

    import timm
    model = timm.create_model(cfg["model"]["backbone"], pretrained=True,
                              num_classes=len(class_names), drop_rate=cfg["model"].get("dropout", 0.3))
    checkpoint = cfg["model"].get("checkpoint")
    if checkpoint and Path(checkpoint).exists():
        try:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            print(f"Loaded checkpoint: {checkpoint}")
        except Exception as e:
            print(f"[WARN] Checkpoint load failed ({e}), starting from pretrained weights")
    model = model.to(device)

    criterion = get_criterion(cfg, cw_tensor, device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    writer = SummaryWriter(exp_dir / "tensorboard")

    patience = tcfg.get("early_stopping_patience", 5)
    min_delta = tcfg.get("early_stopping_min_delta", 0.001)
    best_f1, no_improve = 0.0, 0
    history = []
    t_start = time.time()

    for epoch in range(1, tcfg["epochs"] + 1):
        t_ep = time.time()
        onecycle = scheduler if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else None
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion,
                                      scaler, device, cfg, use_amp, onecycle)
        val_m = evaluate(model, val_loader, criterion, device, class_names, use_amp)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_m["macro_f1"])
            elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        ep_s = time.time() - t_ep
        elapsed = time.time() - t_start
        eta_s = elapsed / epoch * (tcfg["epochs"] - epoch)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Ep {epoch:>3}/{tcfg['epochs']} | "
              f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_m['loss']:.4f} val_f1={val_m['macro_f1']:.4f} | "
              f"lr={lr:.2e} | {ep_s:.0f}s | ETA {eta_s:.0f}s")

        writer.add_scalars("Loss",     {"train": tr_loss, "val": val_m["loss"]}, epoch)
        writer.add_scalars("Accuracy", {"train": tr_acc,  "val": val_m["accuracy"]}, epoch)
        writer.add_scalar("F1/val_macro", val_m["macro_f1"], epoch)
        writer.add_scalar("LR", lr, epoch)

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_m["loss"], "val_acc": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"], "val_weighted_f1": val_m["weighted_f1"],
            "lr": lr, "epoch_time_s": ep_s,
        })

        if val_m["macro_f1"] > best_f1 + min_delta:
            best_f1 = val_m["macro_f1"]
            no_improve = 0
            torch.save(model.state_dict(), exp_dir / "best_model.pth")
            print(f"  ★  New best val F1: {best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    # test eval on best checkpoint
    model.load_state_dict(torch.load(exp_dir / "best_model.pth", weights_only=True))
    test_m = evaluate(model, test_loader, criterion, device, class_names, use_amp)

    print(f"\nTest | acc={test_m['accuracy']:.4f}  macro_f1={test_m['macro_f1']:.4f}")

    metrics = {
        "experiment_name": exp_name,
        "timestamp": ts,
        "backbone": cfg["model"]["backbone"],
        "total_training_time_s": time.time() - t_start,
        "epochs_trained": len(history),
        "best_val_macro_f1": best_f1,
        "test": test_m,
        "history": history,
        "config": cfg,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    writer.close()
    print(f"\nSaved to: {exp_dir}")
    return exp_dir, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_config.yaml")
    args = ap.parse_args()
    main(args.config)
