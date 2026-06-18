# VEGAN: Vision-language and Edge-enhanced GAN for Microscopic Medical Image Segmentation

<p align="center">
  <a href="https://doi.org/10.1109/MECO66322.2025.11049114"><img src="https://img.shields.io/badge/Paper-IEEE%20MECO%202025-blue?style=flat-square&logo=ieee" alt="Paper"></a>
  <a href="https://github.com/souvikpramanik-medimg/VEGAN/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-red?style=flat-square&logo=pytorch" alt="PyTorch">
</p>

<p align="center">
  <b>Gouranga Maity, Souvik Pramanik, Diptarka Mandal, Dmitrii Kaplun, Vyacheslav Gulvanskii, Ram Sarkar</b><br>
  DIAT Pune, DRDO &nbsp;·&nbsp; Jadavpur University &nbsp;·&nbsp; St. Petersburg ETU "LETI"<br><br>
  <i>14th Mediterranean Conference on Embedded Computing (MECO 2025), IEEE</i><br>
  DOI: <a href="https://doi.org/10.1109/MECO66322.2025.11049114">10.1109/MECO66322.2025.11049114</a>
</p>

---

VEGAN combines an Attention U-Net generator with fused edge maps (Sobel + Canny + Laplacian), CLIP-based vision-language embeddings injected into skip connections, and a Pix2Pix GAN with PatchGAN discriminator — achieving state-of-the-art microscopic medical image segmentation without any post-processing.

> **Architecture figure:** see Fig. 1 in the [paper](https://doi.org/10.1109/MECO66322.2025.11049114).

---

## Citation

If you use VEGAN in your research, please cite:

```bibtex
@inproceedings{maity2025vegan,
  title     = {{VEGAN}: A Vision-language and Edge-enhanced {GAN}-based Microscopic Medical Image Segmentation Model},
  author    = {Maity, Gouranga and Pramanik, Souvik and Mandal, Diptarka and Kaplun, Dmitrii and Gulvanskii, Vyacheslav and Sarkar, Ram},
  booktitle = {Proceedings of the 14th Mediterranean Conference on Embedded Computing (MECO)},
  pages     = {1--6},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/MECO66322.2025.11049114}
}
```

---

## Results

### MoNuSeg (Multi-Organ Nuclei Segmentation)

| Method | Dice (%) | IoU (%) |
|--------|----------|---------|
| DCSA-Net (2023) | 73.20 | 58.00 |
| DRI-UNet (2023) | 77.54 | 75.49 |
| A-ReSEUnet (2024) | 77.90 | 63.80 |
| ATTransUNet (2023) | 79.16 | 65.51 |
| DeepFuzz (2023) | 79.10 | 65.42 |
| **VEGAN (Ours)** | **79.24** | **65.68** |

### CNS (Cervical Nuclei Segmentation)

| Method | Dice (%) | IoU (%) |
|--------|----------|---------|
| BTTFA (2019) | 90.00 | 81.82 |
| C-UNet (2023) | 93.12 | 87.13 |
| **VEGAN (Ours)** | **99.56** | **99.12** |

### PH2 (Skin Lesion / Dermoscopy)

| Method | Dice (%) | IoU (%) |
|--------|----------|---------|
| MTL (2024) | 88.81 | 79.89 |
| MEFP-NET (2024) | 91.86 | 85.71 |
| AFCF-Net (2024) | 91.26 | 84.81 |
| **VEGAN (Ours)** | **92.54** | **86.61** |

### Ablation Study

| Configuration | MoNuSeg Dice | MoNuSeg IoU | CNS Dice | CNS IoU |
|---|---|---|---|---|
| Attention U-Net (base) | 72.92 | 57.65 | 99.46 | 98.92 |
| + Edge Map | 73.56 | 58.30 | 99.46 | 98.93 |
| + Edge Map + VLM | 77.51 | 63.32 | 99.52 | 99.04 |
| **+ Edge Map + CBAM + VLM (Full)** | **79.24** | **65.68** | **99.56** | **99.12** |

---

## Installation

```bash
git clone https://github.com/souvikpramanik-medimg/VEGAN.git
cd VEGAN
conda create -n vegan python=3.9 -y
conda activate vegan
pip install -r requirements.txt
```

---

## Datasets

Download and place under `data/`:

| Dataset | Task | Link |
|---------|------|------|
| [MoNuSeg](https://monuseg.grand-challenge.org/) | Multi-organ nuclei | [Grand Challenge](https://monuseg.grand-challenge.org/) |
| [CNS](https://github.com/zhangzjn/CNS) | Cervical nuclei | [GitHub](https://github.com/zhangzjn/CNS) |
| [PH2](https://www.fc.up.pt/addi/ph2%20database.html) | Skin lesion (dermoscopy) | [ADDI Project](https://www.fc.up.pt/addi/ph2%20database.html) |

Expected structure:
```
data/
├── MoNuSeg/
│   ├── train/ {images/, masks/}
│   └── test/  {images/, masks/}
├── CNS/   {images/, masks/}
└── PH2/   {images/, masks/}
```

---

## Training

```bash
python vegan/train.py \
  --image_dir     data/MoNuSeg/train/images \
  --mask_dir      data/MoNuSeg/train/masks \
  --val_image_dir data/MoNuSeg/test/images \
  --val_mask_dir  data/MoNuSeg/test/masks \
  --epochs 200 --batch_size 2 --lr 1e-3
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 200 | Training epochs |
| `--batch_size` | 2 | Batch size |
| `--lr` | 1e-3 | Learning rate (Adam, β=(0.5, 0.999)) |
| `--lambda_dice` | 1.0 | Weight of Dice loss vs adversarial loss |

---

## Inference

```bash
python vegan/inference.py \
  --checkpoint checkpoints/best_generator.pth \
  --input      path/to/image.png \
  --output     results/
```

---

## Project Structure

```
VEGAN/
├── vegan/
│   ├── models/
│   │   ├── attention_unet.py   # Attention U-Net + CBAM generator
│   │   ├── discriminator.py    # PatchGAN discriminator
│   │   └── clip_encoder.py     # CLIP feature extractor
│   ├── utils/
│   │   ├── edge_maps.py        # Sobel + Canny + Laplacian fusion
│   │   ├── losses.py           # Dice loss + adversarial loss
│   │   └── metrics.py          # Dice score, IoU
│   ├── data/
│   │   └── dataset.py          # Dataset loader (MoNuSeg / CNS / PH2)
│   ├── train.py
│   └── inference.py
├── notebooks/
│   └── VEGAN_full_pipeline.ipynb   # Original development notebook (Kaggle)
├── demo.ipynb
├── requirements.txt
├── setup.py
├── CITATION.cff
└── LICENSE
```

---

## Acknowledgements

This work was supported by the CMATER Research Lab, Department of Computer Science and Engineering, Jadavpur University, Kolkata, and the Modelling and Simulation Lab, DIAT Pune, DRDO.

We build on [Attention U-Net](https://arxiv.org/abs/1804.03999), [Pix2Pix](https://arxiv.org/abs/1611.07004), [CLIP](https://arxiv.org/abs/2103.00020), and [CBAM](https://arxiv.org/abs/1807.06521).
