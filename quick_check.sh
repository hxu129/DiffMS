#!/bin/bash
# Quick validation script for DiffMS setup
# Run this after cloning the repository

echo "=========================================="
echo "DiffMS Setup Validation"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "1. Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python3 not found"
    exit 1
fi

# Check PyTorch
echo ""
echo "2. Checking PyTorch..."
python3 -c "import torch; print('   PyTorch version:', torch.__version__); print('   CUDA available:', torch.cuda.is_available())" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PyTorch installed"
else
    echo -e "${RED}✗${NC} PyTorch not found. Install with: pip install torch"
    exit 1
fi

# Check PyTorch Lightning
echo ""
echo "3. Checking PyTorch Lightning..."
python3 -c "import pytorch_lightning; print('   Version:', pytorch_lightning.__version__)" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PyTorch Lightning installed"
else
    echo -e "${RED}✗${NC} PyTorch Lightning not found. Install with: pip install pytorch-lightning"
fi

# Check RDKit
echo ""
echo "4. Checking RDKit..."
python3 -c "import rdkit; from rdkit import Chem; print('   RDKit OK')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} RDKit installed"
else
    echo -e "${RED}✗${NC} RDKit not found. Install with conda: conda install -c conda-forge rdkit"
fi

# Check data directory
echo ""
echo "5. Checking data directory..."
if [ -d "data/canopus" ]; then
    CANOPUS_FILES=$(find data/canopus -name "*.tsv" | wc -l)
    echo -e "${GREEN}✓${NC} CANOPUS data found ($CANOPUS_FILES TSV files)"
else
    echo -e "${YELLOW}⚠${NC} CANOPUS data not found in data/canopus/"
    echo "   Run: bash data_processing/01_download_canopus_data.sh"
fi

if [ -d "data/msg" ]; then
    echo -e "${GREEN}✓${NC} MassSpecGym data found"
else
    echo -e "${YELLOW}⚠${NC} MassSpecGym data not found in data/msg/"
    echo "   Run: bash data_processing/02_download_msg_data.sh"
fi

# Check checkpoints
echo ""
echo "6. Checking model checkpoints..."
if [ -f "checkpoints/diffms_canopus.ckpt" ]; then
    SIZE=$(du -h checkpoints/diffms_canopus.ckpt | cut -f1)
    echo -e "${GREEN}✓${NC} CANOPUS checkpoint found ($SIZE)"
else
    echo -e "${YELLOW}⚠${NC} CANOPUS checkpoint not found: checkpoints/diffms_canopus.ckpt"
    echo "   Download from: https://zenodo.org/records/15122968"
fi

if [ -f "checkpoints/diffms_msg.ckpt" ]; then
    SIZE=$(du -h checkpoints/diffms_msg.ckpt | cut -f1)
    echo -e "${GREEN}✓${NC} MassSpecGym checkpoint found ($SIZE)"
else
    echo -e "${YELLOW}⚠${NC} MassSpecGym checkpoint not found: checkpoints/diffms_msg.ckpt"
    echo "   Download from: https://zenodo.org/records/15122968"
fi

# Check GPU
echo ""
echo "7. Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} GPU found: $GPU_INFO"
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found. GPU may not be available."
fi

# Check configs
echo ""
echo "8. Checking configuration files..."
if [ -f "configs/config.yaml" ]; then
    echo -e "${GREEN}✓${NC} configs/config.yaml found"
else
    echo -e "${RED}✗${NC} configs/config.yaml not found"
fi

if [ -f "configs/dataset/canopus.yaml" ]; then
    echo -e "${GREEN}✓${NC} configs/dataset/canopus.yaml found"
else
    echo -e "${RED}✗${NC} configs/dataset/canopus.yaml not found"
fi

# Summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

# Count issues
ERRORS=0
WARNINGS=0

# Recheck critical items
python3 -c "import torch" 2>/dev/null || ((ERRORS++))
python3 -c "import pytorch_lightning" 2>/dev/null || ((WARNINGS++))
python3 -c "import rdkit" 2>/dev/null || ((ERRORS++))
[ -f "checkpoints/diffms_canopus.ckpt" ] || ((WARNINGS++))
[ -d "data/canopus" ] || ((WARNINGS++))

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "You can now run a quick test with:"
    echo "  cd src"
    echo "  python3 spec2mol_main.py dataset=canopus \\"
    echo "    general.test_only=../checkpoints/diffms_canopus.ckpt \\"
    echo "    general.name=quick_test \\"
    echo "    train.eval_batch_size=5 \\"
    echo "    general.test_samples_to_generate=1 \\"
    echo "    dataset.max_count=5 \\"
    echo "    general.wandb=disabled"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Setup incomplete but runnable${NC}"
    echo "  Errors: $ERRORS"
    echo "  Warnings: $WARNINGS"
    echo ""
    echo "You may need to download data or checkpoints."
else
    echo -e "${RED}✗ Setup incomplete${NC}"
    echo "  Errors: $ERRORS"
    echo "  Warnings: $WARNINGS"
    echo ""
    echo "Please install missing dependencies first."
fi

echo ""
echo "For more information, see README.md"
echo ""

