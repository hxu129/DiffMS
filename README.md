# DiffMS - Diffusion Models for Mass Spectrometry-based Molecule Generation

Official implementation of **DiffMS: A Diffusion Model for De Novo Molecular Generation from Mass Spectra**.

---

## Quick Start Guide

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DiffMS
```

### 2. Environment Setup

#### Option A: Using Conda (Recommended)

```bash
# Create conda environment
conda create -n diffms python=3.9
conda activate diffms

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Verify Existing Environment

If you already have an environment set up, run the validation script:

```bash
# Activate your environment first
conda activate your-env-name  # or diffms

# Run validation script
bash quick_check.sh
```

This will check:
- Python installation
- PyTorch and CUDA availability
- PyTorch Lightning
- RDKit
- Data directories
- Model checkpoints
- GPU availability

Or check manually:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning; print(f'Lightning: {pytorch_lightning.__version__}')"
python -c "import rdkit; print('RDKit: OK')"
```

### 3. Download Data

#### CANOPUS (NPLIB1) Dataset

```bash
# Download from the original source
# Data should be placed in: DiffMS/data/canopus/

# Expected structure:
# data/canopus/
# ├── splits/
# │   └── canopus_hplus_100_0.tsv
# ├── spec_files/
# ├── subformulae/
# │   └── subformulae_default/
# └── labels.tsv
```

#### MassSpecGym Dataset

```bash
# Download from MassSpecGym
# Data should be placed in: DiffMS/data/msg/

# Expected structure:
# data/msg/
# ├── split.tsv
# ├── spec_files/
# ├── subformulae/
# │   └── default_subformulae/
# └── labels.tsv
```

**Note**: Data is already downloaded if you see the `data/` directory populated.

### 4. Download Model Checkpoints

Model checkpoints should be placed in `DiffMS/checkpoints/`:

```bash
# Expected files:
# checkpoints/
# ├── diffms_canopus.ckpt   # CANOPUS model
# ├── diffms_msg.ckpt        # MassSpecGym model
# ├── encoder_canopus.pt     # (optional)
# └── encoder_msg.pt         # (optional)
```

### 5. Quick Functionality Check

Run a quick test to verify everything is working:

```bash
cd src

# Quick test (5 samples, ~2 minutes)
python3 spec2mol_main.py \
  dataset=canopus \
  general.test_only=../checkpoints/diffms_canopus.ckpt \
  general.name=quick_test \
  train.eval_batch_size=5 \
  general.test_samples_to_generate=1 \
  dataset.max_count=5 \
  general.wandb=disabled
```

**Expected output**:
- Process runs for ~2 minutes
- Generates pickle files in `src/preds/`
- Shows test metrics at the end
- No errors

### 6. Run Evaluation

#### Small-scale Test (Recommended First)

Test with 50-100 samples to verify performance (~1-2 hours):

```bash
cd src

python3 spec2mol_main.py \
  dataset=canopus \
  general.test_only=../checkpoints/diffms_canopus.ckpt \
  general.name=canopus_test_100 \
  dataset.max_count=100 \
  general.wandb=disabled
```

#### Full Evaluation on CANOPUS

```bash
cd src

# Single GPU (recommended, ~2.6 days)
python3 spec2mol_main.py \
  dataset=canopus \
  general.test_only=../checkpoints/diffms_canopus.ckpt \
  general.name=canopus_full \
  general.wandb=disabled

# Multi-GPU (if available, ~1.3 days with 2 GPUs)
python3 spec2mol_main.py \
  dataset=canopus \
  general.test_only=../checkpoints/diffms_canopus.ckpt \
  general.name=canopus_full_2gpu \
  general.gpus=2 \
  general.wandb=disabled
```

#### Full Evaluation on MassSpecGym

```bash
cd src

python3 spec2mol_main.py \
  dataset=msg \
  general.test_only=../checkpoints/diffms_msg.ckpt \
  general.name=msg_full \
  model.encoder_hidden_dim=512 \
  general.wandb=disabled
```

**Note**: MassSpecGym requires `model.encoder_hidden_dim=512` due to checkpoint architecture.

### 7. Monitor Progress

#### Using tmux (Recommended for long runs)

```bash
# Start a tmux session
tmux new-session -s diffms_eval

# Inside tmux, run evaluation
cd /path/to/DiffMS/src
python3 spec2mol_main.py ...

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t diffms_eval
```

#### Check logs

```bash
# Find latest run
ls -lt src/outputs/

# View log
tail -f src/outputs/YYYY-MM-DD/HH-MM-SS-{name}/spec2mol_main.log
```

#### Monitor GPU

```bash
# Real-time GPU monitoring
nvidia-smi dmon -c 100

# Check GPU usage
nvidia-smi
```

### 8. Results and Analysis

After evaluation completes, results are saved in:

```
src/outputs/YYYY-MM-DD/HH-MM-SS-{name}/
├── spec2mol_main.log           # Main log file
├── preds/
│   ├── {name}_rank_0_pred_*.pkl  # Generated molecules
│   └── {name}_rank_0_true_*.pkl  # Ground truth molecules
└── logs/
    └── {name}/
        └── version_0/
            └── metrics.csv        # Evaluation metrics
```

## Configuration Options

### Key Parameters

```yaml
# GPU settings
general.gpus: 1              # Number of GPUs (1, 2, or -1 for all)

# Evaluation settings
general.test_samples_to_generate: 100  # Molecules per spectrum
train.eval_batch_size: 128             # Batch size for evaluation
dataset.max_count: null                # Limit dataset size (null for full)

# Model settings
model.encoder_hidden_dim: 256          # Encoder dimension (512 for MSG)

# Logging
general.wandb: disabled                # WandB logging (online/offline/disabled)
```

### Example Configurations

**Fast test** (10 minutes):
```bash
general.test_samples_to_generate=10 dataset.max_count=50
```

**Balanced test** (2 hours):
```bash
general.test_samples_to_generate=10 dataset.max_count=100
```

**Full paper reproduction** (2-3 days):
```bash
# Use default settings (no overrides needed)
```

## Performance Expectations

### Speed (Single NVIDIA RTX A6000)

| Configuration | Time per Molecule | Full Dataset (819 samples) |
|---------------|-------------------|---------------------------|
| batch_size=1 | ~12 seconds | ~11 days |
| batch_size=5+ | ~2.7 seconds | **~2.6 days** |

**Key insight**: Larger batch sizes dramatically improve efficiency!

### Expected Metrics (CANOPUS Dataset)

Based on paper Table 1:

| Metric | Top-1 | Top-10 |
|--------|-------|--------|
| Accuracy | ~17% | ~33% |
| Tanimoto Similarity | ~0.36 | ~0.59 |
| Validity | >95% | >95% |

**Note**: Small-scale tests (< 50 samples) may show high variance.

## Troubleshooting

### Issue: "FileNotFoundError: data/..."

**Solution**: Ensure data paths in config files use absolute paths:
```bash
# Check and update if needed
vim configs/dataset/canopus.yaml
# Set: datadir: '/absolute/path/to/DiffMS/data/canopus'
```

### Issue: "KeyError: pytorch-lightning_version"

**Solution**: This is expected for our checkpoints. The code handles it automatically.

### Issue: "wandb.errors.UsageError: api_key not configured"

**Solution**: Disable WandB:
```bash
general.wandb=disabled
```

### Issue: Slow performance

**Solutions**:
1. Ensure batch size is large (≥8): `train.eval_batch_size=128`
2. Use multiple GPUs: `general.gpus=2`
3. Reduce generation count for testing: `general.test_samples_to_generate=10`

### Issue: Out of memory

**Solutions**:
1. Reduce batch size: `train.eval_batch_size=64`
2. Reduce generation count: `general.test_samples_to_generate=50`

## Code Structure

```
DiffMS/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── dataset/               # Dataset configs
│   │   ├── canopus.yaml
│   │   └── msg.yaml
│   └── general/               # General settings
│       └── general_default.yaml
├── src/                       # Source code
│   ├── spec2mol_main.py      # Main evaluation script
│   ├── diffusion_model_spec2mol.py  # DiffMS model
│   ├── datasets/             # Dataset loaders
│   └── metrics/              # Evaluation metrics
├── data/                      # Data directory
├── checkpoints/              # Model checkpoints
└── README.md
```

## Key Files Modified from Original

- `src/diffusion_model_spec2mol.py`: Added progress logging in `test_step`
- Dataset configs: Updated to use absolute paths

## Citation

If you use this code, please cite:

```bibtex
@article{diffms2024,
  title={DiffMS: A Diffusion Model for De Novo Molecular Generation from Mass Spectra},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Additional Documentation

- `CURRENT_STATUS.md` - Current implementation status
- `FINAL_SUMMARY.md` - Complete usage guide and troubleshooting
- `PERFORMANCE_BREAKTHROUGH.md` - Performance optimization details
- `TEST_RESULTS_ANALYSIS.md` - How to interpret evaluation results

## Support

For issues:
1. Check existing documentation files
2. Verify configuration with quick test
3. Check logs in `src/outputs/`

---

**Last Updated**: 2025-10-17  
**Tested On**: NVIDIA RTX A6000, PyTorch 2.x, Python 3.9
