"""
=============================================================================
NEU-DET — Test & Inference Script
=============================================================================
Kullanım:
  1) Tüm validation seti üzerinde test:
       python test.py

  2) Tek bir görüntü üzerinde tahmin:
       python test.py --image "NEU-DET/validation/images/crazing/crazing_241.jpg"

  3) Belirli bir model ile test (varsayılan: resnet18):
       python test.py --model resnet18
       python test.py --model custom_cnn
=============================================================================
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ── Ayarlar ───────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(SCRIPT_DIR, "NEU-DET")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CLASSES     = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled-in_scale", "scratches"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Model tanımları (deeplearning.py ile aynı) ────────────────────────────
class CustomCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
            )
        self.features   = nn.Sequential(conv_block(3,32), conv_block(32,64),
                                         conv_block(64,128), conv_block(128,256))
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256,512),
                                         nn.ReLU(True), nn.Dropout(0.5), nn.Linear(512,num_classes))
    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


def build_resnet18(num_classes=6):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256), nn.ReLU(True),
        nn.Dropout(0.4), nn.Linear(256, num_classes))
    return model


def load_model(model_name: str):
    """Kaydedilmiş .pth dosyasından modeli yükler."""
    if model_name == "resnet18":
        model     = build_resnet18(NUM_CLASSES)
        pth_path  = os.path.join(RESULTS_DIR, "resnet18_best.pth")
        label     = "ResNet18 Transfer Learning"
    else:
        model     = CustomCNN(NUM_CLASSES)
        pth_path  = os.path.join(RESULTS_DIR, "custom_cnn_best.pth")
        label     = "Custom CNN"

    if not os.path.exists(pth_path):
        print(f"  HATA: {pth_path} bulunamadı!")
        print(f"  Önce 'python deeplearning.py' çalıştırarak modeli eğitin.")
        sys.exit(1)

    state = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"  Model yüklendi: {label}")
    print(f"  Weights: {pth_path}")
    return model, label


# ── 1) Validation seti üzerinde tam test ─────────────────────────────────
def test_full_validation(model, label):
    val_dir = os.path.join(DATA_ROOT, "validation", "images")
    dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            preds = model(inputs.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc        = accuracy_score(all_labels, all_preds)

    print(f"\n{'='*58}")
    print(f"  {label} — Validation Test Sonuçları")
    print(f"{'='*58}")
    print(f"  Toplam örnek : {len(all_labels)}")
    print(f"  Doğru tahmin : {(all_preds==all_labels).sum()}")
    print(f"  Accuracy     : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n  Per-Class Raporu:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Confusion matrix kaydet
    cm       = confusion_matrix(all_labels, all_preds)
    out_path = os.path.join(RESULTS_DIR, f"test_cm_{label.replace(' ','_')}.png")
    fig, ax  = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{label} — Test Confusion Matrix  (Acc={acc:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix kaydedildi: {out_path}")
    return acc


# ── 2) Tek görüntü tahmini ────────────────────────────────────────────────
def predict_single(model, label, image_path: str):
    if not os.path.exists(image_path):
        print(f"  HATA: Görüntü bulunamadı: {image_path}")
        sys.exit(1)

    img      = Image.open(image_path).convert("RGB")
    tensor   = val_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = probs.argmax()

    print(f"\n{'='*50}")
    print(f"  Görüntü   : {os.path.basename(image_path)}")
    print(f"  Model     : {label}")
    print(f"  Tahmin    : {CLASSES[pred]}  ({probs[pred]*100:.1f}%)")
    print(f"\n  Tüm sınıf olasılıkları:")
    for i, (cls, p) in enumerate(zip(CLASSES, probs)):
        bar   = "█" * int(p * 30)
        arrow = " ← TAHMİN" if i == pred else ""
        print(f"    {cls:<18} {p*100:5.1f}%  {bar}{arrow}")
    print(f"{'='*50}")

    # Görüntüyü + olasılıkları kaydet
    out_path = os.path.join(RESULTS_DIR, "single_prediction.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    inv = transforms.Normalize(mean=[-0.485/0.229,-0.456/0.224,-0.406/0.225],
                                std =[1/0.229, 1/0.224, 1/0.225])
    img_show = inv(val_transform(img)).permute(1,2,0).clamp(0,1).numpy()
    axes[0].imshow(img_show, cmap="gray")
    axes[0].set_title(f"Tahmin: {CLASSES[pred]}\nGüven: {probs[pred]*100:.1f}%",
                      fontsize=12, fontweight="bold",
                      color="green" if CLASSES[pred] in image_path else "blue")
    axes[0].axis("off")
    colors = ["#2196F3" if i != pred else "#4CAF50" for i in range(NUM_CLASSES)]
    axes[1].barh(CLASSES, probs * 100, color=colors)
    axes[1].set_xlabel("Olasılık (%)")
    axes[1].set_title("Sınıf Olasılıkları")
    axes[1].set_xlim(0, 110)
    for i, p in enumerate(probs):
        axes[1].text(p*100 + 1, i, f"{p*100:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafik kaydedildi: {out_path}")


# ── Ana giriş ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEU-DET Model Test")
    parser.add_argument("--model", choices=["resnet18","custom_cnn"],
                        default="resnet18", help="Kullanılacak model")
    parser.add_argument("--image", type=str, default=None,
                        help="Tek görüntü tahmini için dosya yolu")
    args = parser.parse_args()

    print(f"\n{'='*58}")
    print(f"  NEU-DET Test Script  |  Device: {DEVICE}")
    print(f"{'='*58}")

    model, label = load_model(args.model)

    if args.image:
        predict_single(model, label, args.image)
    else:
        test_full_validation(model, label)
