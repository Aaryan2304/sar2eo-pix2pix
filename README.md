# Sar2EO: SAR-to-EO Image Translation

Pix2Pix (conditional GAN) implementation for translating Sentinel-1 SAR (VV) patches into Sentinel-2 EO (RGB) patches. This is the submission for GalaxEye's Satellite AI Research Intern assessment.

## Approach

- **Model**: Pix2Pix (Isola et al., 2017) — U-Net generator (55M params) + PatchGAN discriminator (2.8M params)
- **Loss**: L1 reconstruction (λ=100) + adversarial (BCE) for the full model; L1-only for the ablation baseline
- **Data**: SEN1-2 dataset (Sentinel-1&2 Image Pairs, Kaggle/TUM) — 12,800 train / 1,600 val / 1,600 test, split by terrain class
- **Preprocessing**: SAR and EO normalised to [-1, 1]; no augmentation

## Requirements

- Python 3.10
- PyTorch 2.6.0+cu124 (CUDA 12.4 compatible)
- See `requirements.txt` for full list

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Place the Kaggle Sentinel-1&2 Image Pairs dataset in `./dataset/` with this structure:

```
dataset/
├── agri/
│   ├── s1/   (SAR VV, single-channel 8-bit PNG)
│   └── s2/   (Optical RGB, 3-channel 8-bit PNG)
├── barrenland/
├── grassland/
└── urban/
```

The dataset comes with a `splits.json` file that defines train/val/test splits by terrain class to avoid near-duplicate leakage.

## Training

```bash
# Experiment A: L1-only baseline (no adversarial loss)
python train.py --config config.yaml --experiment exp_A_l1_only

# Experiment B: Full Pix2Pix (L1 + adversarial loss)
python train.py --config config.yaml --experiment exp_B_l1_gan
```

Training logs per-epoch losses to CSV and saves loss curve plots. Checkpoints saved every 10 epochs.

## Inference

```bash
python infer.py --input_dir ./test_sar/ --output_dir ./test_eo/ --weights ./outputs/exp_B_l1_gan/checkpoints/gen_final.pth
```

Input: directory of single-channel 256×256 8-bit PNG SAR patches (VV, dB-scaled, [0,255]).
Output: directory of 256×256 RGB PNG images with matching filenames.

## Evaluation

```bash
python eval.py --config config.yaml --weights ./outputs/exp_B_l1_gan/checkpoints/gen_final.pth --experiment exp_B_l1_gan
```

Computes LPIPS, FID, SSIM, PSNR on both val and test splits. Generates qualitative triplets (SAR → Generated → GT).

## Project Structure

```
├── config.yaml              # All hyperparameters
├── train.py                 # Training script
├── infer.py                 # Inference (I/O contract compliant)
├── eval.py                  # Metrics + qualitative results
├── requirements.txt         # Pinned dependencies
├── models/
│   └── __init__.py          # U-Net generator + PatchGAN discriminator
├── data/
│   └── __init__.py          # PyTorch Dataset + dataloader
├── utils/
│   ├── metrics.py           # LPIPS, FID, SSIM, PSNR
│   └── visualization.py     # Qualitative triplet generation
└── outputs/
    ├── loss_curves/         # Plots and raw CSV
    ├── qualitative/         # SAR → Generated → GT triplets
    └── checkpoints/         # Saved model weights
```

## Key Design Decisions

1. **Pix2Pix over CycleGAN**: We have paired data — Pix2Pix exploits this directly. CycleGAN is designed for unpaired translation and would waste the pairing information.

2. **Pix2Pix over diffusion**: Diffusion models give marginally better perceptual quality but require 2-5× more compute. With the 13-day timeline and free-tier GPU constraints, Pix2Pix gives reliable results with time for proper evaluation.

3. **Ablation**: L1-only vs L1+GAN cleanly isolates the discriminator's contribution to perceptual quality.

4. **Split by terrain class**: Adjacent patches from the same scene are near-duplicates. Splitting by class (agriculture, barrenland, grassland, urban) ensures val/test contain different geographies.

## Model Weights

The trained Pix2Pix generator is hosted on Hugging Face Hub:

**https://huggingface.co/NeuralNomad0101/sar2eo-pix2pix**

Includes a model card with usage example and full performance metrics.

## Results

| Metric | Exp A (L1-only, test) | Exp B (Pix2Pix, test) | Δ |
|--------|----------------------|----------------------|---|
| LPIPS ↓ | 0.5822 | **0.3848** | **−34%** |
| FID ↓ | 166.31 | **96.33** | **−42%** |
| SSIM ↑ | **0.4246** | 0.2740 | −35% |
| PSNR ↑ (dB) | **20.18** | 17.59 | −13% |

The adversarial loss improves perceptual quality (LPIPS, FID) at the expected cost of pixel-level metrics (SSIM, PSNR).

## References

- Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (Pix2Pix), CVPR 2017
- Schmitt et al., "SEN1-2: A dataset of Sentinel-1 and Sentinel-2 image pairs", 2018
- Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (LPIPS), CVPR 2018
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (FID), NeurIPS 2017
