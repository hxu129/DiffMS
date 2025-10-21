# DiffMS Setup Checklist

After cloning the repository, follow this checklist:

## ☐ 1. Environment Setup

```bash
# Create conda environment
conda create -n diffms python=3.9
conda activate diffms

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -e .
```

**Verify**:
```bash
bash quick_check.sh
```

## ☐ 2. Download Data

### CANOPUS (NPLIB1)
```bash
bash data_processing/01_download_canopus_data.sh
```

**Expected**: `data/canopus/` with splits/, spec_files/, labels.tsv

### MassSpecGym
```bash
bash data_processing/02_download_msg_data.sh
```

**Expected**: `data/msg/` with split.tsv, spec_files/, labels.tsv

## ☐ 3. Download Model Checkpoints

Download from: https://zenodo.org/records/15122968

Place files in `checkpoints/`:
- `diffms_canopus.ckpt`
- `diffms_msg.ckpt`
- `encoder_canopus.pt` (optional)
- `encoder_msg.pt` (optional)

## ☐ 4. Fix Config Paths (if needed)

Update to absolute paths:

```bash
# Edit dataset configs
vim configs/dataset/canopus.yaml
vim configs/dataset/msg.yaml

# Change relative paths like '../data/canopus' to:
# /absolute/path/to/DiffMS/data/canopus
```

## ☐ 5. Run Quick Test

```
python3 spec2mol_main.py dataset=canopus general.test_only=/local3/ericjiang/wgc/huaxu/ms/DiffMS/checkpoints/diffms_canopus.ckpt general.name=canopus_small_test train.eval_batch_size=8 general.test_samples_to_generate=10 general.wandb=disabled dataset.max_count=50
```

```bash
cd src
python3 spec2mol_main.py \
  dataset=canopus \
  general.test_only=../checkpoints/diffms_canopus.ckpt \
  general.name=quick_test \
  train.eval_batch_size=5 \
  general.test_samples_to_generate=1 \
  dataset.max_count=5 \
  general.wandb=disabled
```

**Expected**:
- Runs for ~2 minutes
- No errors
- Creates `src/preds/quick_test_*.pkl` files
- Shows test metrics

## ☐ 6. Run Full Evaluation (Optional)

### Small test (100 samples, ~1-2 hours)
```bash
cd src
python3 spec2mol_main.py \
  dataset=canopus \
  general.test_only=../checkpoints/diffms_canopus.ckpt \
  general.name=canopus_100 \
  dataset.max_count=100 \
  general.wandb=disabled
```

### Full test (819 samples, ~2.6 days)
```bash
cd src
python3 spec2mol_main.py \
  dataset=canopus \
  general.test_only=../checkpoints/diffms_canopus.ckpt \
  general.name=canopus_full \
  general.wandb=disabled
```

## Troubleshooting

### Issue: ImportError for torch/rdkit/etc
**Solution**: Make sure conda environment is activated:
```bash
conda activate diffms
```

### Issue: FileNotFoundError for data
**Solution**: Check data paths in configs are absolute:
```bash
pwd  # Get current directory
# Update configs/dataset/*.yaml with this path
```

### Issue: Checkpoint loading error
**Solution**: This is expected for our checkpoints, code handles it automatically.

### Issue: "wandb" error
**Solution**: Add `general.wandb=disabled` to command

## Quick Reference Commands

```bash
# Activate environment
conda activate diffms

# Check setup
bash quick_check.sh

# Quick test
cd src && python3 spec2mol_main.py dataset=canopus \
  general.test_only=../checkpoints/diffms_canopus.ckpt \
  general.name=test general.wandb=disabled \
  dataset.max_count=5 train.eval_batch_size=5 \
  general.test_samples_to_generate=1

# Monitor GPU
nvidia-smi dmon

# Check logs
tail -f src/outputs/*/spec2mol_main.log
```

---

**Need help?** See `README.md` for detailed instructions or `FINAL_SUMMARY.md` for troubleshooting.
