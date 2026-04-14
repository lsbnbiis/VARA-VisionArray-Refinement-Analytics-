# 🧠 VARA (VisionArray Refinement Analytics), an autoencoder-based Defect Detection for TPMI-ARRAY (9-Grid High-Resolution Images)

This project implements an **unsupervised anomaly detection pipeline** using a **Autoencoder (PyTorch)** for **high-resolution grayscale images (7000×7000)**.

It is designed for industrial defect detection scenarios where:

- Only **normal samples are abundant**
- Defects are **rare or hard to annotate**
- Images are **very large (7000×7000)**

---

## 🚀 Key Features

- ✅ Handles **7000×7000 grayscale images**
- ✅ Splits each image into **9 overlapping patches (3×3 grid)**
- ✅ Patch size: **3550×3550**
- ✅ Uses **Autoencoder reconstruction error** for anomaly detection
- ✅ Supports:
  - Patch-level scoring
  - Image-level classification
- ✅ Automatic **threshold selection using validation set**

---



## 📦 Installation

```bash
pip install torch torchvision scikit-learn pillow tqdm numpy
```

---

## 🧩 Method Overview

### 1. 9-Grid Splitting

Each image is split into 9 overlapping patches:

```
Original: 7000 × 7000
Patch size: 3550 × 3550
Stride: 1725
```

---

### 2. Training Strategy

- Train Autoencoder using **only normal images**
- Model learns to reconstruct normal patterns
- Defects → higher reconstruction error

---

### 3. Inference Strategy

1. Split image into 9 patches  
2. Compute reconstruction error per patch  
3. Compare with threshold:

```
score > threshold → defect
```

---

## 🏋️ Training

```bash
python train.py \
  --data_dir data \
  --patch_size 3550 \
  --input_size 256 \
  --batch_size 8 \
  --epochs 30 \
  --lr 1e-3 \
  --save_dir checkpoints
```

---

## 📌 Training Notes

- Training only uses:
  ```
  data/train/normal/
  ```
- Validation uses:
  ```
  data/val/normal/
  data/val/defect/
  ```
- Automatically computes:
  - ROC-AUC
  - PR-AUC
  - Best threshold (F1-based)

---

## 💾 Checkpoints

```
checkpoints/
├── ae_best.pt
├── ae_last.pt
```

Each checkpoint contains:

- Model weights
- Optimizer state
- Threshold
- Training config
- Validation metrics

---

## 🔍 Inference

### Single Image

```bash
python inference.py \
  --checkpoint checkpoints/ae_best.pt \
  --input path/to/image.png
```

---

### Folder Inference

```bash
python inference.py \
  --checkpoint checkpoints/ae_best.pt \
  --input path/to/folder \
  
```


---



## ⚠️ Practical Tips



### 1. Recommended Setup

| Scenario | Suggested Config |
|--------|----------------|
| Small defects | input_size=512 |
| Large defects | input_size=256 |
| Noisy data | agg=top2 |



---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

- PyTorch
- scikit-learn
- Industrial anomaly detection research
