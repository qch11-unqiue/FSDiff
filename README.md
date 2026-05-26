# PI-FSDiff: SAR RFI Mitigation with Frequency-Spatial Conditional Diffusion

PI-FSDiff is a PyTorch implementation of a physics-informed frequency-spatial conditional diffusion framework for image-domain synthetic aperture radar (SAR) radio frequency interference (RFI) mitigation.

The model restores RFI-contaminated SAR amplitude images by combining conditional DDPM restoration, coarse RFI mask guidance, frequency-domain global interaction, Strip-Masked Image Modeling pretraining, and frequency-domain consistency regularization.

Code: https://github.com/qch11-unqiue/FSDiff

## Highlights

- **Conditional diffusion for SAR RFI mitigation**: the denoising network takes the noisy diffusion state, the RFI-contaminated SAR image, and a coarse RFI mask as conditional input.
- **FSAS bottleneck module**: a Frequency Spatial Attention System is embedded in the U-Net bottleneck to model long-range stripe-like interference with frequency-domain interaction.
- **Strip-MIM pretraining**: stripe-shaped random masks simulate SAR RFI morphology and provide task-aware restoration priors before fine-tuning.
- **Frequency-domain loss**: FFT consistency regularizes the reverse-estimated clean image to preserve SAR texture and high-frequency structures.
- **Ablation-ready code**: independent scripts are provided for removing FSAS, FFT loss, MIM pretraining, mask weighting, mask condition, and release sampling.

## Project Structure

```text
.
|-- data/
|   `-- dataset.py                  # SAR image, RFI image, and mask dataset loader
|-- diffusion/
|   `-- ddpm.py                     # conditional DDPM process and mask-guided sampling
|-- model/
|   |-- unet.py                     # diffusion U-Net backbone
|   |-- fsas.py                     # Frequency Spatial Attention System
|-- ablation_experiments/
|   |-- README.md                   # ablation-specific instructions
|   |-- train_*.py                  # ablation training entry points
|   `-- infer_*.py                  # ablation inference entry points
|-- train_mim_new.py                # Strip-MIM pretraining
|-- train.py                        # PI-FSDiff fine-tuning
|-- inference.py                    # batch inference and metric calculation
|-- mask2pt.py                      # mask conversion utility
|-- metrics.py                      # DRSR, entropy, PSNR, SSIM, MAE, RMSE, etc.
```

## Environment

The code is tested with Python and PyTorch. A CUDA-enabled GPU is recommended because diffusion sampling is iterative.

Install the main dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy pillow tqdm einops
```

Depending on your CUDA version, install PyTorch from the official PyTorch command generator if the command above is not suitable for your machine.

## Data Preparation

The default training scripts use hardcoded paths. Before running, edit the path variables in the corresponding script.

For fine-tuning, `RealSARDataset` expects the following layout:

```text
<DATASET_ROOT>/
`-- image/
    |-- clean/
    |   |-- 0001.png
    |   `-- ...
    `-- rfi/
        |-- 0001.png
        `-- ...
```

The mask directory is loaded separately and is expected to contain `.pt` masks with names matching the image stems:

```text
<MASK_DIR>/
|-- 0001.pt
`-- ...
```

For inference, prepare:

```text
<INPUT_RFI_FOLDER>/       # RFI-contaminated SAR images
<INPUT_MASK_FOLDER>/      # .pt or .png masks with matching file names
<INPUT_CLEAN_FOLDER>/     # optional clean references for full-reference metrics
<OUTPUT_FOLDER>/          # restored output images
```

Images are loaded as single-channel grayscale SAR amplitude patches and resized to `512 x 512` by default.

## Usage

### 1. Strip-MIM Pretraining

Edit these variables in `train_mim_new.py`:

```python
TRAIN_DATA_PATH = "/path/to/clean/sar/patches"
SAVE_DIR = "/path/to/save/mim/checkpoints"
```

Then run:

```bash
python train_mim_new.py
```

The script saves checkpoints such as:

```text
mim_pretrained_epoch_300.pth
checkpoint_latest.pth
```

### 2. Fine-Tuning PI-FSDiff

Edit these variables in `train.py`:

```python
DATASET_ROOT = "/path/to/dataset/root"
SAVE_DIR = "/path/to/save/fine_tuned/checkpoints"
MIM_WEIGHT_PATH = "/path/to/mim_pretrained_epoch_300.pth"
```

Then run:

```bash
python train.py
```

The default fine-tuning configuration uses:

```text
image size: 512
batch size: 8
epochs: 200
diffusion steps: 1000
learning rate: 1e-4
FFT loss weight: 0.05
```

### 3. Inference

Edit these variables in `inference.py`:

```python
CHECKPOINT_PATH = "/path/to/unet_epoch_199.pth"
INPUT_RFI_FOLDER = "/path/to/rfi/images"
INPUT_MASK_FOLDER = "/path/to/rfi/masks"
INPUT_CLEAN_FOLDER = "/path/to/clean/references"  # optional
OUTPUT_FOLDER = "/path/to/save/restored/images"
```

Then run:

```bash
python inference.py
```

The script saves restored images to `OUTPUT_FOLDER` and reports SAR-oriented metrics such as DRSR and entropy. If clean references are provided, it also reports PSNR, SSIM, MAE, and RMSE.

## Ablation Experiments

Ablation code is located in `ablation_experiments/`.

First edit:

```text
ablation_experiments/hardcoded_config.py
```

Then run the desired entry script:

```bash
python ablation_experiments/train_full.py
python ablation_experiments/train_no_fsas.py
python ablation_experiments/train_no_fft.py
python ablation_experiments/train_no_mim_pretrain.py
python ablation_experiments/train_no_mask_weight.py
python ablation_experiments/train_no_mask_condition.py
```

Inference scripts are also provided:

```bash
python ablation_experiments/infer_full.py
python ablation_experiments/infer_no_fsas.py
python ablation_experiments/infer_no_fft.py
python ablation_experiments/infer_no_mim_pretrain.py
python ablation_experiments/infer_no_mask_weight.py
python ablation_experiments/infer_no_mask_condition.py
python ablation_experiments/infer_no_release.py
```

See `ablation_experiments/README.md` for the full ablation table and direct-run instructions.

## Model Input and Output

During fine-tuning and inference, the denoiser input is formed as:

```text
[noisy_diffusion_state, rfi_contaminated_image, rfi_mask]
```

For single-channel SAR images, this gives a 3-channel input tensor. The U-Net predicts the diffusion noise, and the DDPM reverse process produces the restored SAR image.

## Metrics

The repository includes:

- `DRSR`: directional RFI suppression ratio for stripe-related spectral suppression.
- `Entropy`: auxiliary SAR-oriented statistic.
- `PSNR`, `SSIM`, `MAE`, `RMSE`: full-reference metrics for paired or semi-simulated evaluation.
- Legacy compatibility metrics such as `ENL` and `EPI`.

## Notes

- Most scripts currently use hardcoded paths for convenient direct execution in PyCharm or a terminal. Update the path configuration before training or inference.
- The released implementation is image-domain and uses single-channel SAR amplitude images. It does not take complex SLC phase data as model input.
- Mask files can be produced by external RFI segmentation methods, traditional detectors, or saved manually as `.pt` or `.png` files.
- DDPM inference with 1000 steps is computationally heavier than deterministic CNN restoration. It is more suitable for offline high-fidelity restoration.
- `model/evs.py` is kept as an optional experimental module and may require `mamba-ssm`; the default U-Net uses `model/fsas.py` and does not require it.

## Citation

If this code is helpful for your research, please cite the corresponding paper when it becomes available:

```bibtex
@article{pifsdiff,
  title   = {PI-FSDiff: Stripe-Masked Frequency-Spatial Conditional Diffusion for SAR Radio Frequency Interference Mitigation},
  author  = {Hu, Canbin and Qu, Caihai and Sun, Xiaokun},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026}
}
```

Project repository: https://github.com/qch11-unqiue/FSDiff

## License

Please add a license file before public release. Common choices include MIT, Apache-2.0, and GPL-3.0. If the code is intended only for academic research, state the usage restriction clearly in `LICENSE`.
