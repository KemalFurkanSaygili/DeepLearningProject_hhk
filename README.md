# NEU-DET Steel Surface Defect Classification — CNN Project

## Problem Definition
Automated classification of steel surface defects using **Convolutional Neural Networks (CNN)**.  
The goal is to correctly classify grayscale images of steel surfaces into one of **6 defect categories**.

## Dataset
**NEU-DET** (NEU Surface Defect Database)

| Split      | Images per Class | Total |
|------------|:---------------:|:-----:|
| Train      |       240       | 1 440 |
| Validation |        60       |   360 |

**Classes:**
| #  | Class Name       | Description                       |
|----|------------------|-----------------------------------|
| 1  | `crazing`        | Network of fine cracks on surface |
| 2  | `inclusion`      | Embedded foreign particles        |
| 3  | `patches`        | Irregular surface patches         |
| 4  | `pitted_surface` | Pits/holes on the surface         |
| 5  | `rolled-in_scale`| Scale pressed into the surface    |
| 6  | `scratches`      | Linear surface scratches          |

## Models Implemented

### 1. Custom CNN
A 4-block convolutional network built from scratch:
```
Conv Block 1 : 3  → 32  channels (Conv2d × 2, BN, ReLU, MaxPool)
Conv Block 2 : 32 → 64  channels
Conv Block 3 : 64 → 128 channels
Conv Block 4 : 128→ 256 channels
Global Average Pooling
FC: 256 → 512 → Dropout(0.5) → 6
```

### 2. ResNet18 (Transfer Learning)
Pretrained on ImageNet, final FC layer replaced and fine-tuned:
```
ResNet18 (frozen) → FC: 512 → 256 → ReLU → Dropout(0.4) → 6
```

## How to Run

### Requirements
```bash
python -m pip install torch torchvision matplotlib scikit-learn seaborn pdfplumber
```

### Full Training (1440 train images)
```bash
python deeplearning.py
```

### Submission Mode (100 samples only)
Open `deeplearning.py` and set:
```python
SUBMISSION_MODE = True
```
Then run:
```bash
python deeplearning.py
```

## Output Files (`results/` folder)

| File | Description |
|------|-------------|
| `dataset_distribution.png` | Class distribution bar chart |
| `training_history.png`     | Loss & accuracy curves for both models |
| `cm_custom_cnn.png`        | Confusion matrix — Custom CNN |
| `cm_resnet18.png`          | Confusion matrix — ResNet18 |
| `samples_custom_cnn.png`   | Sample prediction grid — Custom CNN |
| `samples_resnet18.png`     | Sample prediction grid — ResNet18 |
| `model_comparison.png`     | Accuracy & F1 comparison bar chart |
| `custom_cnn_best.pth`      | Best Custom CNN weights |
| `resnet18_best.pth`        | Best ResNet18 weights |

## Configuration

Edit these constants at the top of `deeplearning.py`:

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `NUM_EPOCHS` | 20 | Number of training epochs |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `LR` | 1e-3 | Initial learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `IMG_SIZE` | 224 | Input image size |
| `SUBMISSION_MODE` | False | Limit dataset to 100 samples |
