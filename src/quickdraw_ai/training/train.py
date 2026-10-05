import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from quickdraw_ai.data.dataset import QuickDrawDataset
from quickdraw_ai.models.cnn import SketchCNN


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply small random affine changes without changing sketch identity."""
    batch = images.shape[0]
    device = images.device
    angles = (torch.rand(batch, device=device) * 20.0 - 10.0) * (np.pi / 180.0)
    scales = 0.9 + torch.rand(batch, device=device) * 0.2
    cos, sin = torch.cos(angles) * scales, torch.sin(angles) * scales
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    theta[:, :, 2] = torch.rand(batch, 2, device=device) * 0.16 - 0.08
    grid = functional.affine_grid(theta, images.shape, align_corners=False)
    return functional.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QuickDraw sketch classifier")
    parser.add_argument("--data", default="data")
    parser.add_argument("--classes", default="config/classes.txt")
    parser.add_argument("--output", default="models/quickdraw.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-per-class", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots", default="demo/training_curves.png")
    parser.add_argument("--no-augment", action="store_true", help="disable random rotation/translation augmentation")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    classes = [line.strip() for line in Path(args.classes).read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = QuickDrawDataset(args.data, classes, args.max_per_class)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.1,
                                          random_state=args.seed, stratify=dataset.labels)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SketchCNN(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_top3_accuracy": []}
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if not args.no_augment:
                images = augment_batch(images)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        model.eval()
        val_losses: list[float] = []
        targets: list[int] = []
        predictions: list[int] = []
        top3_predictions: list[list[int]] = []
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                val_losses.append(loss_fn(logits, labels.to(device)).item() * len(labels))
                targets.extend(labels.tolist())
                predictions.extend(logits.argmax(1).cpu().tolist())
                top3_predictions.extend(logits.topk(min(3, len(classes)), dim=1).indices.cpu().tolist())
        accuracy = accuracy_score(targets, predictions)
        top3_accuracy = sum(target in top for target, top in zip(targets, top3_predictions)) / len(targets)
        train_loss = total_loss / len(train_idx)
        val_loss = sum(val_losses) / len(val_idx)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(accuracy)
        history["val_top3_accuracy"].append(top3_accuracy)
        print(f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"accuracy={accuracy:.4f} top3={top3_accuracy:.4f}")
        if accuracy > best_acc:
            best_acc = accuracy
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.cpu().state_dict(), "classes": classes,
                        "image_size": 28, "epoch": epoch + 1, "metrics": {
                            "accuracy": accuracy,
                            "top3_accuracy": top3_accuracy,
                            "precision_macro": precision_score(targets, predictions, average="macro", zero_division=0),
                            "recall_macro": recall_score(targets, predictions, average="macro", zero_division=0),
                            "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0),
                        }}, output)
            model.to(device)
    if args.plots:
        plot_path = Path(args.plots)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="validation")
        axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
        axes[0].legend()
        axes[1].plot(history["val_accuracy"], label="accuracy")
        axes[1].plot(history["val_top3_accuracy"], label="top-3 accuracy")
        axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)
        print(f"training curves: {plot_path}")
    print(f"best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
