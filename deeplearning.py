"""
=============================================================================
NEU-DET Steel Surface Defect Classification using CNN
=============================================================================
Course   : Deep Learning
Dataset  : NEU-DET (NEU Surface Defect Database)
Model    : Custom CNN + Transfer Learning (ResNet18)
Author   : Kemal Furkan Saygili
=============================================================================

Dataset Structure:
  NEU-DET/
    train/images/{crazing, inclusion, patches, pitted_surface,
                  rolled-in_scale, scratches}/  -> 240 images each (1440 total)
    validation/images/{same 6 classes}/         -> 60 images each  (360 total)

How to Run:
  1. Make sure NEU-DET folder is in the same directory as this script.
  2. Install requirements:
       python -m pip install torch torchvision matplotlib scikit-learn seaborn
  3. Run:
       python deeplearning.py
  4. Results (plots + model) will be saved in ./results/ folder.
=============================================================================
"""

import os
import copy
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(SCRIPT_DIR, "NEU-DET")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES      = ["crazing", "inclusion", "patches",
                "pitted_surface", "rolled-in_scale", "scratches"]
NUM_CLASSES  = len(CLASSES)
IMG_SIZE     = 224          # ResNet input size
BATCH_SIZE   = 32
NUM_EPOCHS   = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SUBMISSION_MODE     = False   # Set True to use only 100 samples
SUBMISSION_SAMPLES  = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  NEU-DET Steel Defect Classification — CNN Project")
print(f"{'='*60}")
print(f"  Device  : {DEVICE}")
print(f"  Results : {RESULTS_DIR}")
print(f"{'='*60}\n")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

train_dir = os.path.join(DATA_ROOT, "train", "images")
val_dir   = os.path.join(DATA_ROOT, "validation", "images")

full_train = datasets.ImageFolder(train_dir, transform=train_transform)
full_val   = datasets.ImageFolder(val_dir,   transform=val_transform)

if SUBMISSION_MODE:
    per_class = SUBMISSION_SAMPLES // NUM_CLASSES
    indices = []
    for cls_idx in range(NUM_CLASSES):
        cls_indices = [i for i, (_, l) in enumerate(full_train.samples) if l == cls_idx]
        indices.extend(random.sample(cls_indices, min(per_class, len(cls_indices))))
    train_dataset = Subset(full_train, indices)
    val_dataset   = full_val
    print(f"  SUBMISSION MODE: {len(train_dataset)} train / {len(val_dataset)} val samples\n")
else:
    train_dataset = full_train
    val_dataset   = full_val
    print(f"  Dataset: {len(train_dataset)} train / {len(val_dataset)} val samples")
    print(f"  Classes: {CLASSES}\n")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

class CustomCNN(nn.Module):
    """
    A compact but expressive CNN built for 6-class defect classification.
    Architecture:
        Conv Block 1 : 3  → 32  (3×3, BN, ReLU, MaxPool)
        Conv Block 2 : 32 → 64  (3×3, BN, ReLU, MaxPool)
        Conv Block 3 : 64 → 128 (3×3, BN, ReLU, MaxPool)
        Conv Block 4 : 128→ 256 (3×3, BN, ReLU, MaxPool)
        Global Avg Pool
        FC  → 512 → Dropout(0.5) → 6
    """
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            conv_block(3,   32),   # 224→112
            conv_block(32,  64),   # 112→56
            conv_block(64,  128),  # 56→28
            conv_block(128, 256),  # 28→14
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # 14→1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes=6):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def train_model(model, loaders, criterion, optimizer, scheduler,
                num_epochs, model_name="model"):
    """
    Trains a model and returns:
        best_model_state  — state dict with best val accuracy
        history           — dict of train/val loss & acc lists
    """
    model.to(DEVICE)
    best_state  = copy.deepcopy(model.state_dict())
    best_acc    = 0.0
    history     = {"train_loss": [], "val_loss": [],
                   "train_acc":  [], "val_acc":  []}

    print(f"\n  Training [{model_name}] — {num_epochs} epochs")
    print(f"  {'-'*50}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            loader = loaders[phase]

            running_loss, running_correct = 0.0, 0
            total_samples = 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                total_samples   += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc  = running_correct / total_samples

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc   = epoch_acc
                    best_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>2}/{num_epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f}  Acc: {history['train_acc'][-1]:.4f} | "
              f"Val Loss: {history['val_loss'][-1]:.4f}  Acc: {history['val_acc'][-1]:.4f} | "
              f"Time: {elapsed:.1f}s")

    print(f"\n  Best Val Accuracy [{model_name}]: {best_acc:.4f}")
    return best_state, history


def evaluate(model, loader):
    """Returns (all_preds, all_labels) numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            preds  = model(inputs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_training_history(histories, names, save_path):
    """Side-by-side loss & accuracy curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, (h, name) in enumerate(zip(histories, names)):
        c = colors[i % len(colors)]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], "--",  color=c, alpha=0.6, label=f"{name} Train")
        axes[0].plot(epochs, h["val_loss"],   "-",   color=c,            label=f"{name} Val")
        axes[1].plot(epochs, h["train_acc"],  "--",  color=c, alpha=0.6, label=f"{name} Train")
        axes[1].plot(epochs, h["val_acc"],    "-",   color=c,            label=f"{name} Val")

    for ax, title, ylabel in zip(axes,
                                  ["Loss Curve", "Accuracy Curve"],
                                  ["Loss", "Accuracy"]):
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training History — NEU-DET Defect Classification",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title(title,        fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_sample_predictions(model, loader, class_names, title, save_path,
                            n_samples=12):
    """Show a grid of sample images with true / predicted labels."""
    model.eval()
    images_shown, true_labels, pred_labels = [], [], []
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std =[1/0.229,       1/0.224,       1/0.225])

    with torch.no_grad():
        for inputs, labels in loader:
            preds = model(inputs.to(DEVICE)).argmax(dim=1).cpu()
            for img, lbl, pred in zip(inputs, labels, preds):
                images_shown.append(inv_norm(img).permute(1, 2, 0).clamp(0, 1).numpy())
                true_labels.append(lbl.item())
                pred_labels.append(pred.item())
                if len(images_shown) >= n_samples:
                    break
            if len(images_shown) >= n_samples:
                break

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i in range(n_samples):
        axes[i].imshow(images_shown[i], cmap="gray")
        correct = true_labels[i] == pred_labels[i]
        color   = "green" if correct else "red"
        axes[i].set_title(f"T: {class_names[true_labels[i]]}\n"
                          f"P: {class_names[pred_labels[i]]}",
                          fontsize=8, color=color, fontweight="bold")
        axes[i].axis("off")
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_class_distribution(save_path):
    """Bar chart of train vs val class distribution."""
    train_counts = [240] * NUM_CLASSES
    val_counts   = [60]  * NUM_CLASSES
    x = np.arange(NUM_CLASSES)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, train_counts, width, label="Train",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, val_counts,   width, label="Validation",
                   color="#FF5722", alpha=0.85)

    ax.set_title("Dataset Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=9)
    ax.set_ylabel("Number of Images")
    ax.legend()
    ax.bar_label(bars1, padding=3, fontsize=8)
    ax.bar_label(bars2, padding=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results, save_path):
    """Bar chart comparing accuracy & F1 of both models."""
    model_names = list(results.keys())
    accs = [results[m]["accuracy"] for m in model_names]
    f1s  = [results[m]["f1"]       for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#4CAF50", alpha=0.85)
    b2 = ax.bar(x + width/2, f1s,  width, label="F1-Score (macro)", color="#9C27B0", alpha=0.85)
    ax.set_title("Model Comparison — Validation Set", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=3)
    ax.bar_label(b2, fmt="%.3f", padding=3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    loaders = {"train": train_loader, "val": val_loader}
    criterion = nn.CrossEntropyLoss()
    results   = {}
    histories = []
    names     = []
    print("\n[1/6] Plotting dataset distribution...")
    plot_class_distribution(os.path.join(RESULTS_DIR, "dataset_distribution.png"))
    print("\n[2/6] Building Custom CNN...")
    custom_cnn = CustomCNN(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in custom_cnn.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer_cnn = optim.Adam(custom_cnn.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cnn = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cnn, mode="min", factor=0.5, patience=3)

    best_state_cnn, history_cnn = train_model(
        custom_cnn, loaders, criterion, optimizer_cnn, scheduler_cnn,
        NUM_EPOCHS, model_name="CustomCNN")

    custom_cnn.load_state_dict(best_state_cnn)
    torch.save(best_state_cnn, os.path.join(RESULTS_DIR, "custom_cnn_best.pth"))

    preds_cnn, labels_cnn = evaluate(custom_cnn, val_loader)
    acc_cnn = accuracy_score(labels_cnn, preds_cnn)
    f1_cnn  = f1_score(labels_cnn, preds_cnn, average="macro")
    cm_cnn  = confusion_matrix(labels_cnn, preds_cnn)
    results["Custom CNN"] = {"accuracy": acc_cnn, "f1": f1_cnn}
    histories.append(history_cnn)
    names.append("Custom CNN")

    print(f"\n  Custom CNN — Validation Report:")
    print(classification_report(labels_cnn, preds_cnn, target_names=CLASSES))

    plot_confusion_matrix(cm_cnn, CLASSES,
                          "Custom CNN — Confusion Matrix",
                          os.path.join(RESULTS_DIR, "cm_custom_cnn.png"))
    plot_sample_predictions(custom_cnn, val_loader, CLASSES,
                            "Custom CNN — Sample Predictions",
                            os.path.join(RESULTS_DIR, "samples_custom_cnn.png"))
    print("\n[3/6] Building ResNet18 (Transfer Learning)...")
    resnet18 = build_resnet18(num_classes=NUM_CLASSES)
    trainable = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in resnet18.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")

    optimizer_res = optim.Adam(filter(lambda p: p.requires_grad, resnet18.parameters()),
                               lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_res = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_res, mode="min", factor=0.5, patience=3)

    best_state_res, history_res = train_model(
        resnet18, loaders, criterion, optimizer_res, scheduler_res,
        NUM_EPOCHS, model_name="ResNet18")

    resnet18.load_state_dict(best_state_res)
    torch.save(best_state_res, os.path.join(RESULTS_DIR, "resnet18_best.pth"))

    preds_res, labels_res = evaluate(resnet18, val_loader)
    acc_res = accuracy_score(labels_res, preds_res)
    f1_res  = f1_score(labels_res, preds_res, average="macro")
    cm_res  = confusion_matrix(labels_res, preds_res)
    results["ResNet18 TL"] = {"accuracy": acc_res, "f1": f1_res}
    histories.append(history_res)
    names.append("ResNet18 TL")

    print(f"\n  ResNet18 — Validation Report:")
    print(classification_report(labels_res, preds_res, target_names=CLASSES))

    plot_confusion_matrix(cm_res, CLASSES,
                          "ResNet18 Transfer Learning — Confusion Matrix",
                          os.path.join(RESULTS_DIR, "cm_resnet18.png"))
    plot_sample_predictions(resnet18, val_loader, CLASSES,
                            "ResNet18 TL — Sample Predictions",
                            os.path.join(RESULTS_DIR, "samples_resnet18.png"))
    print("\n[4/6] Saving training history plot...")
    plot_training_history(histories, names,
                          os.path.join(RESULTS_DIR, "training_history.png"))

    print("\n[5/6] Saving model comparison plot...")
    plot_model_comparison(results,
                          os.path.join(RESULTS_DIR, "model_comparison.png"))
    print("\n[6/6] Summary")
    print(f"  {'='*50}")
    print(f"  {'Model':<18} {'Val Accuracy':>14} {'F1 Macro':>12}")
    print(f"  {'-'*50}")
    for name, res in results.items():
        print(f"  {name:<18} {res['accuracy']:>14.4f} {res['f1']:>12.4f}")
    print(f"  {'='*50}")

    best_model = max(results, key=lambda m: results[m]["accuracy"])
    print(f"\n  Best model: {best_model}  "
          f"(Acc={results[best_model]['accuracy']:.4f})")
    print(f"\n  All results saved to: {RESULTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
