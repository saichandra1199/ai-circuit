import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets


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
    label_ids = list(range(len(class_names)))
    report = classification_report(all_labels, all_preds, target_names=class_names,
                                   labels=label_ids, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {c: report[c] for c in class_names},
        "confusion_matrix": cm,
    }


def evaluate_checkpoint(checkpoint_path: str, config_path_or_dict, split: str = "test") -> dict:
    """Load checkpoint and evaluate on a data split. Returns metrics dict."""
    if isinstance(config_path_or_dict, dict):
        cfg = config_path_or_dict
    else:
        with open(config_path_or_dict) as f:
            cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    with open(cfg["paths"].get("class_mapping") or data_dir / "class_mapping.json") as f:
        class_mapping = json.load(f)
    with open(cfg["paths"].get("class_weights") or data_dir / "class_weights.json") as f:
        cw_dict = json.load(f)

    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    cw_tensor = torch.tensor([cw_dict[c] for c in class_names], dtype=torch.float)

    from train import build_transforms
    _, val_tf = build_transforms(cfg)
    ds = datasets.ImageFolder(data_dir / split, transform=val_tf)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"] * 2,
                        shuffle=False, num_workers=cfg["training"].get("num_workers", 0),
                        pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"

    import timm
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone = ckpt["backbone"] if isinstance(ckpt, dict) and "backbone" in ckpt else cfg["model"].get("backbone", "efficientnet_b0")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(class_names))
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)

    loss_cfg = cfg.get("loss", {})
    w = cw_tensor.to(device) if loss_cfg.get("use_class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=loss_cfg.get("label_smoothing", 0.0))

    metrics = evaluate(model, loader, criterion, device, class_names, use_amp)
    print(f"[evaluate] {split} | acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on test/val/train split")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="training_config.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--output", help="Optional path to save metrics JSON")
    args = ap.parse_args()

    metrics = evaluate_checkpoint(args.checkpoint, args.config, args.split)

    print(f"\nBackbone: {args.checkpoint}")
    print(f"Split:    {args.split}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nPer-class:")
    for cls, vals in metrics["per_class"].items():
        print(f"  {cls:<25} f1={vals['f1-score']:.4f}  prec={vals['precision']:.4f}  rec={vals['recall']:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved → {args.output}")

    return metrics


if __name__ == "__main__":
    main()
