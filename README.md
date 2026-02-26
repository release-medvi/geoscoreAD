# GeoScoreAD

Official implementation of **GeoScoreAD** for few-shot anomaly detection.

This repository provides a minimal and reproducible pipeline for:

- AnomalyCLIP baseline
- AdaCLIP baseline
- Few-shot semantic MLP refinement
- Pixel-level and image-level evaluation (AUROC, AUPR, F1, PRO, AUPRO)

---

## 1. Environment Setup

### Requirements

- Python 3.10+
- PyTorch
- torchvision
- numpy
- scikit-image
- scikit-learn
- pandas
- omegaconf
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Environment Variables (Recommended)

Before running experiments, set the following environment variables.

### Dataset Root (Required)
Set the dataset root directory using an environment variable:

```bash
export DATA_ROOT=/path/to/datasets
```

### Model Cache Directory (Optional)

By default, pretrained CLIP models are cached under:

```bash
~/.cache/clip
```
If you are running on a cluster or want to specify a custom cache location, set:

```bash
export CACHE_DIR=/path/to/cache
```


## 3. Pretrained Weights

AdaCLIP / AnomalyCLIP pretrained weights are not included.
Download official pretrained weights and place them under:

```bash
checkpoints/
```

## 4. License
This repository uses external components from:

- AnomalyCLIP
- AdaCLIP

Please follow their respective licenses.
