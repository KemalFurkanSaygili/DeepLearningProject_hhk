"""
Generate NEU-DET Steel Defect Classification Report (.docx)
Run: python generate_report.py
Output: steel_defect_report.docx
"""
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_DIR    = os.path.join(SCRIPT_DIR, os.pardir, "NEU-DET", "train", "images")
OUT_PATH    = os.path.join(SCRIPT_DIR, "steel_defect_report.docx")

CLASSES = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]

def img(name): return os.path.join(RESULTS_DIR, name)
def sample(cls, n=1): return os.path.join(DATA_DIR, cls, f"{cls}_{n}.jpg")

doc = Document()
for s in doc.sections:
    s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Cm(2.5)

sty = doc.styles["Normal"]
sty.font.name = "Arial"
sty.font.size = Pt(12)

def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in h.runs:
        run.font.name = "Arial"
        run.font.color.rgb = RGBColor(0,0,0)
        run.font.size = Pt(14 if level==1 else 12)
        run.bold = True
    return h

def para(text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    r = p.add_run(text)
    r.font.name = "Arial"; r.font.size = Pt(12)
    r.bold = bold; r.italic = italic
    return p

def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.name = "Arial"; r.font.size = Pt(12)

def figure(path, caption, width=Inches(5.2)):
    if not os.path.exists(path):
        print(f"  SKIP: {path}")
        return
    doc.add_picture(path, width=width)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.runs[0].font.name = "Arial"; c.runs[0].font.size = Pt(10); c.runs[0].italic = True

def table_grid(headers, rows_data):
    t = doc.add_table(rows=1+len(rows_data), cols=len(headers))
    t.style = "Table Grid"
    for i,h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = h
        c.paragraphs[0].runs[0].bold = True
        c.paragraphs[0].runs[0].font.name = "Arial"
    for ri, row_data in enumerate(rows_data):
        for ci, val in enumerate(row_data):
            c = t.rows[ri+1].cells[ci]; c.text = val
            c.paragraphs[0].runs[0].font.name = "Arial"
            c.paragraphs[0].runs[0].font.size = Pt(11)
    return t

# ── TITLE PAGE ────────────────────────────────────────────────────────────────
doc.add_paragraph(); doc.add_paragraph()

tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("DEEP LEARNING\nPROJECT REPORT")
r.font.name = "Arial"; r.font.size = Pt(18); r.bold = True

doc.add_paragraph()
tp2 = doc.add_paragraph(); tp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = tp2.add_run("Steel Surface Defect Classification\nUsing CNN and ResNet18 Transfer Learning\non the NEU-DET Dataset")
r2.font.name = "Arial"; r2.font.size = Pt(14); r2.bold = True

doc.add_paragraph(); doc.add_paragraph()

auth = doc.add_paragraph(); auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
ra = auth.add_run("Prepared by:\nAd Soyad 1\nAd Soyad 2\nAd Soyad 3")
ra.font.name = "Arial"; ra.font.size = Pt(12)

doc.add_paragraph()
dp = doc.add_paragraph(); dp.alignment = WD_ALIGN_PARAGRAPH.CENTER
rd = dp.add_run("May 2025")
rd.font.name = "Arial"; rd.font.size = Pt(12)

doc.add_page_break()

# ── 1. ABSTRACT ───────────────────────────────────────────────────────────────
heading("Abstract")
para(
    "This report presents a deep learning based approach for classifying steel surface defects "
    "using the NEU-DET (Northeastern University Surface Defect Database) dataset. Six types of "
    "defects — crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches — "
    "are identified using two CNN-based models: a Custom CNN trained from scratch and a "
    "ResNet18 model adapted via Transfer Learning. Both models were trained for 20 epochs and "
    "evaluated on a held-out validation set. The ResNet18 achieved 98.61% accuracy and the "
    "Custom CNN achieved 97.78%, demonstrating the effectiveness of deep learning for "
    "automated industrial quality control.")

doc.add_paragraph()

# ── 2. INTRODUCTION ───────────────────────────────────────────────────────────
heading("1. Introduction")
para(
    "Steel surface defects represent a significant challenge in the manufacturing industry. "
    "Manual inspection is slow, inconsistent, and prone to human error. Automated vision-based "
    "systems using deep learning can detect defects with high accuracy at production speeds. "
    "Convolutional Neural Networks (CNNs) are the dominant architecture for image classification "
    "tasks due to their ability to learn hierarchical spatial features directly from raw pixels.")
para(
    "This project implements and compares two CNN-based approaches: (1) a Custom CNN designed "
    "from scratch, and (2) ResNet18 with Transfer Learning from ImageNet. The goal is to "
    "classify six distinct steel surface defect types and understand why these two approaches "
    "yield different performance levels.")
doc.add_paragraph()

# ── 3. DATASET ────────────────────────────────────────────────────────────────
heading("2. Dataset — NEU-DET")
para(
    "The NEU-DET dataset is a standard benchmark for steel surface defect detection, "
    "provided by Northeastern University, China. It contains 1,800 grayscale images "
    "across 6 defect classes, with each image sized 200×200 pixels.")

doc.add_paragraph()
table_grid(
    ["Defect Class", "Train Images", "Val Images", "Description"],
    [
        ("Crazing",         "240", "60", "Network of fine surface cracks"),
        ("Inclusion",       "240", "60", "Embedded foreign particles"),
        ("Patches",         "240", "60", "Irregular discolored areas"),
        ("Pitted Surface",  "240", "60", "Small pits or holes"),
        ("Rolled-in Scale", "240", "60", "Scale pressed into surface"),
        ("Scratches",       "240", "60", "Linear grooves on surface"),
        ("TOTAL",           "1440","360","Balanced dataset"),
    ]
)
doc.add_paragraph()

para(
    "The dataset is perfectly balanced: each class has exactly 240 training images and "
    "60 validation images. Images are grayscale (stored as RGB), and the visual similarity "
    "between some classes (e.g., pitted surface vs. scratches) makes this a non-trivial "
    "classification task.")

figure(img("dataset_distribution.png"),
       "Figure 1. Class distribution of the NEU-DET dataset (train and validation splits).")
doc.add_paragraph()

# ── DATASET SAMPLES ───────────────────────────────────────────────────────────
heading("2.1 Dataset Sample Images", level=2)
para(
    "Below are representative sample images from each defect class. Note the visual "
    "differences and similarities that challenge the classification models.")

for i, cls in enumerate(CLASSES):
    sp = sample(cls, 1)
    figure(sp, f"Figure {i+2}. Sample image — class: '{cls}'", width=Inches(2.5))

doc.add_paragraph()

# Data augmentation
heading("2.2 Data Augmentation", level=2)
para("The following augmentation techniques were applied to the training set only:")
for aug in [
    "Random horizontal flip (p=0.5)",
    "Random vertical flip (p=0.5)",
    "Random rotation ±15°",
    "Color jitter: brightness ±20%, contrast ±20%, saturation ±10%",
    "Resize to 224×224 (required for ResNet18 compatibility)",
    "Normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] (ImageNet stats)",
]:
    bullet(aug)
doc.add_paragraph()

# ── 4. MODEL ARCHITECTURES ────────────────────────────────────────────────────
heading("3. Model Architectures")

heading("3.1 Custom CNN", level=2)
para(
    "The Custom CNN was designed specifically for this 6-class defect classification task. "
    "It uses a progressive feature extraction strategy with four convolutional blocks, "
    "each doubling the number of feature maps while halving the spatial dimensions:")

for item in [
    "Conv Block 1: 3→32 filters  | Conv2D(3×3)×2 → BatchNorm → ReLU → MaxPool(2×2) | Output: 112×112",
    "Conv Block 2: 32→64 filters | Conv2D(3×3)×2 → BatchNorm → ReLU → MaxPool(2×2) | Output: 56×56",
    "Conv Block 3: 64→128 filters| Conv2D(3×3)×2 → BatchNorm → ReLU → MaxPool(2×2) | Output: 28×28",
    "Conv Block 4: 128→256 filters| Conv2D(3×3)×2 → BatchNorm → ReLU → MaxPool(2×2)| Output: 14×14",
    "Global Average Pooling → 256-dim feature vector",
    "Fully Connected: 256 → 512 → ReLU → Dropout(0.5) → 6 (softmax)",
]:
    bullet(item)

para(
    "Total trainable parameters: ~2.4 million. "
    "BatchNormalization after each Conv layer stabilizes training and allows "
    "higher learning rates. Dropout(0.5) in the classifier head prevents overfitting.")
doc.add_paragraph()

heading("3.2 ResNet18 with Transfer Learning", level=2)
para(
    "ResNet18 is an 18-layer deep residual network pre-trained on ImageNet "
    "(1.2 million images, 1000 classes). Its key innovation is the residual (skip) "
    "connection: instead of learning H(x), each block learns F(x) = H(x) − x, "
    "making it easier to optimize. This solves the vanishing gradient problem and "
    "allows training of much deeper networks.")
para(
    "For this project, all backbone layers were frozen and only the classification "
    "head was replaced:")
for item in [
    "Backbone: ResNet18 (conv1 → layer1 → layer2 → layer3 → layer4 → AvgPool) — FROZEN",
    "Feature dimension: 512 (ResNet18 output)",
    "New head: Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→6)",
    "Trainable parameters: 132,870 (~1.2% of total 11.2M parameters)",
]:
    bullet(item)
doc.add_paragraph()

# ── 5. WHY DO THEY GIVE DIFFERENT RESULTS? ────────────────────────────────────
heading("4. Why Do ResNet18 and Custom CNN Give Different Results?")
para(
    "This is the central analytical question of this project. Despite the Custom CNN "
    "having ~18× more trainable parameters, ResNet18 outperforms it. The reasons "
    "are explained below:")

heading("4.1 Pre-trained Feature Representations", level=2)
para(
    "ResNet18 was trained on 1.2 million images spanning 1000 categories. During this "
    "training, it learned to detect universal visual features: edges, textures, corners, "
    "gradients, and complex patterns. These low-level features are directly applicable "
    "to steel defect images. The Custom CNN, trained on only 1,440 images, must learn "
    "all these features from scratch — a much harder optimization problem.")

heading("4.2 Network Depth and Residual Connections", level=2)
para(
    "ResNet18 has 18 layers with skip connections. The Custom CNN has 8 convolutional "
    "layers (no skip connections). Residual connections allow gradients to flow directly "
    "from deep layers back to shallow layers, enabling effective training of deeper "
    "representations. Without them, the Custom CNN cannot build as rich a hierarchy "
    "of features, limiting its discriminative power for visually similar classes.")

heading("4.3 Training Data Efficiency", level=2)
para(
    "Transfer Learning dramatically reduces the data requirement. ResNet18 needs only "
    "to learn a 132K-parameter classification head on top of already-powerful features. "
    "The Custom CNN must optimize 2.4M parameters with the same 1,440 training images "
    "(240 per class), making it prone to overfitting despite augmentation and Dropout.")

heading("4.4 Most Confused Classes", level=2)
para(
    "Both confusion matrices reveal that the hardest class pairs are 'pitted_surface' "
    "and 'scratches'. These defects share similar low-level texture features "
    "(sharp, localized discontinuities). ResNet18's richer feature space — built from "
    "millions of images — provides better discrimination. The Custom CNN's simpler "
    "feature hierarchy struggles more with these borderline cases.")
doc.add_paragraph()

# ── 6. TRAINING SETUP ─────────────────────────────────────────────────────────
heading("5. Training Configuration")
table_grid(
    ["Hyperparameter", "Value"],
    [
        ("Optimizer",        "Adam"),
        ("Learning Rate",    "1e-3"),
        ("Weight Decay",     "1e-4"),
        ("Loss Function",    "Cross-Entropy Loss"),
        ("LR Scheduler",     "ReduceLROnPlateau (factor=0.5, patience=3)"),
        ("Batch Size",       "32"),
        ("Epochs",           "20"),
        ("Image Size",       "224×224 pixels"),
        ("Device",           "CPU / CUDA (auto-detected)"),
        ("Seed",             "42 (reproducibility)"),
    ]
)
doc.add_paragraph()

# ── 7. RESULTS ────────────────────────────────────────────────────────────────
heading("6. Simulation Results")

heading("6.1 Training History", level=2)
para(
    "Both models were trained for 20 epochs. The ReduceLROnPlateau scheduler "
    "automatically reduced the learning rate when validation loss stopped improving, "
    "helping both models converge to better optima.")
figure(img("training_history.png"),
       "Figure 8. Loss and accuracy curves for Custom CNN and ResNet18 over 20 epochs.")
doc.add_paragraph()

heading("6.2 Custom CNN Results", level=2)
para("Performance on the 360-image validation set:")
for item in [
    "Validation Accuracy: 97.78%  (352/360 correct)",
    "Macro F1-Score: 0.9777",
    "Perfect classification (F1=1.00): crazing, patches, rolled-in_scale",
    "Most errors: pitted_surface ↔ scratches confusion",
]:
    bullet(item)
figure(img("cm_custom_cnn.png"),
       "Figure 9. Confusion matrix — Custom CNN on validation set.")
figure(img("samples_custom_cnn.png"),
       "Figure 10. Sample Custom CNN predictions (green = correct, red = incorrect).")
doc.add_paragraph()

heading("6.3 ResNet18 Transfer Learning Results", level=2)
para("Performance on the 360-image validation set:")
for item in [
    "Validation Accuracy: 98.61%  (355/360 correct)",
    "Macro F1-Score: 0.9861",
    "Correctly classified: 355 out of 360 images",
    "Fewer errors on pitted_surface and scratches compared to Custom CNN",
]:
    bullet(item)
figure(img("cm_resnet18.png"),
       "Figure 11. Confusion matrix — ResNet18 Transfer Learning on validation set.")
figure(img("samples_resnet18.png"),
       "Figure 12. Sample ResNet18 predictions (green = correct, red = incorrect).")
doc.add_paragraph()

heading("6.4 Model Comparison", level=2)
figure(img("model_comparison.png"),
       "Figure 13. Validation accuracy and macro F1-score comparison.")
doc.add_paragraph()
table_grid(
    ["Model", "Val Accuracy", "Macro F1", "Trainable Params", "Advantage"],
    [
        ("Custom CNN",      "97.78%", "0.9777", "~2.4M",   "No pre-training needed"),
        ("ResNet18 (TL)",   "98.61%", "0.9861", "132,870", "Faster, more accurate"),
    ]
)
doc.add_paragraph()

# ── 8. DISCUSSION ─────────────────────────────────────────────────────────────
heading("7. Discussion and Analysis")
para(
    "Both models achieve excellent accuracy above 97%, confirming that CNN-based "
    "architectures are well suited for industrial surface defect classification. "
    "However, the performance gap — though small in percentage — is meaningful in "
    "industrial applications where every misclassification has a cost.")
para(
    "The key insight is that Transfer Learning provides a better accuracy/cost tradeoff: "
    "ResNet18 achieves higher accuracy while training 18× fewer parameters. This is "
    "because the ImageNet-pretrained backbone already encodes rich, generalizable "
    "visual features. The Custom CNN must discover all features from 1,440 images, "
    "which is a fundamentally harder optimization problem.")
para(
    "Both confusion matrices show that 'pitted_surface' and 'scratches' are the most "
    "commonly confused pair. Visual inspection of samples confirms they share similar "
    "textures: both appear as sharp, localized surface discontinuities with varying "
    "spatial distributions. ResNet18 handles this pair better, likely because its "
    "deeper feature hierarchy captures more subtle distributional differences.")
para(
    "Data augmentation (random flips, rotation, color jitter) was critical for "
    "preventing overfitting in the Custom CNN given the small dataset size. "
    "The ReduceLROnPlateau scheduler provided adaptive learning rate control, "
    "ensuring both models converged without getting stuck in poor local minima.")
doc.add_paragraph()

# ── 9. CONCLUSION ─────────────────────────────────────────────────────────────
heading("8. Conclusion")
para(
    "This project successfully applied deep learning to the classification of six steel "
    "surface defect types from the NEU-DET dataset. Two architectures were compared:")
for item in [
    "Custom CNN (trained from scratch): 97.78% validation accuracy, 2.4M parameters",
    "ResNet18 with Transfer Learning: 98.61% validation accuracy, 132,870 parameters",
]:
    bullet(item)
para(
    "Transfer Learning with ResNet18 proved superior: higher accuracy, fewer trainable "
    "parameters, and faster convergence. The pre-trained ImageNet features generalized "
    "effectively to the steel defect domain. Both models are viable for industrial "
    "deployment, with ResNet18 being the recommended choice.")
para(
    "Future directions include: (1) fine-tuning ResNet18 backbone with low LR, "
    "(2) experimenting with EfficientNet or Vision Transformers, "
    "(3) applying Grad-CAM to visualize which image regions drive predictions, "
    "and (4) testing on larger, more diverse steel defect datasets.")
doc.add_paragraph()

# ── 10. HOW TO RUN ────────────────────────────────────────────────────────────
heading("9. How to Run the Code")

heading("Installation", level=2)
cp = doc.add_paragraph()
r = cp.add_run("python -m pip install torch torchvision matplotlib scikit-learn seaborn python-docx")
r.font.name = "Courier New"; r.font.size = Pt(10)

heading("Train both models", level=2)
c2 = doc.add_paragraph()
r2 = c2.add_run("python deeplearning.py")
r2.font.name = "Courier New"; r2.font.size = Pt(10)

heading("Test with a single image", level=2)
c3 = doc.add_paragraph()
r3 = c3.add_run(
    'python test.py --model resnet18 --image "path/to/your/image.jpg"\n'
    'python test.py --model custom_cnn --image "path/to/your/image.jpg"')
r3.font.name = "Courier New"; r3.font.size = Pt(10)

heading("Run full validation test", level=2)
c4 = doc.add_paragraph()
r4 = c4.add_run("python test.py --model resnet18")
r4.font.name = "Courier New"; r4.font.size = Pt(10)

doc.add_paragraph()

# ── REFERENCES ────────────────────────────────────────────────────────────────
heading("References")
refs = [
    'K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for '
    'hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.',
    'He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. '
    'CVPR 2016, pp. 770-778.',
    'NEU Surface Defect Database. http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html',
    'Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS 32.',
    'Pan, S. J., & Yang, Q. (2009). A survey on transfer learning. '
    'IEEE Trans. Knowledge and Data Engineering, 22(10), 1345-1359.',
    'LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.',
]
for ref in refs:
    p = doc.add_paragraph(style="List Number")
    r = p.add_run(ref)
    r.font.name = "Arial"; r.font.size = Pt(11)

# ── SAVE ──────────────────────────────────────────────────────────────────────
doc.save(OUT_PATH)
print(f"\nReport saved: {OUT_PATH}")
