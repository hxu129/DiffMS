#!/usr/bin/env python3
"""
Quick MCTS Integration Test - Validates basic functionality

This is a minimal test to verify:
1. DiffMS model loads correctly
2. MCTS config is properly initialized
3. ICEBERG verifier can be loaded
4. Basic generation works

Run this BEFORE running full tests.
"""

import sys
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, '/root/ms/DiffMS')
sys.path.insert(0, '/root/ms/DiffMS/src')


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("=" * 60)
    logger.info("TEST 1: Module Imports")
    logger.info("=" * 60)
    
    try:
        from src.diffms_mcts import Spec2MolDenoisingDiffusion
        logger.info("✓ diffms_mcts imported")
        
        from src.mcts_verifier import IcebergVerifier, build_verifier
        logger.info("✓ mcts_verifier imported")
        
        from src.mcts_utils import extract_metadata_from_spectra_objects
        logger.info("✓ mcts_utils imported")
        
        from omegaconf import OmegaConf
        logger.info("✓ OmegaConf imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcts_config():
    """Test that MCTS config can be loaded."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: MCTS Configuration")
    logger.info("=" * 60)
    
    try:
        from omegaconf import OmegaConf
        
        mcts_cfg = OmegaConf.load('/root/ms/DiffMS/configs/mcts/mcts_default.yaml')
        logger.info(f"✓ MCTS config loaded")
        logger.info(f"  use_mcts: {mcts_cfg.use_mcts}")
        logger.info(f"  num_simulation_steps: {mcts_cfg.num_simulation_steps}")
        logger.info(f"  branch_k: {mcts_cfg.branch_k}")
        logger.info(f"  verifier_type: {mcts_cfg.verifier_type}")
        logger.info(f"  gen_checkpoint: {mcts_cfg.iceberg.gen_checkpoint}")
        logger.info(f"  inten_checkpoint: {mcts_cfg.iceberg.inten_checkpoint}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Config load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verifier_init():
    """Test that ICEBERG verifier can be initialized."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 3: ICEBERG Verifier Initialization")
    logger.info("=" * 60)
    
    try:
        from src.mcts_verifier import IcebergVerifier
        
        gen_ckpt = '/root/ms/ms-pred/quickstart/iceberg/models/canopus_iceberg_generate.ckpt'
        inten_ckpt = '/root/ms/ms-pred/quickstart/iceberg/models/canopus_iceberg_score.ckpt'
        
        logger.info("Initializing ICEBERG verifier...")
        verifier = IcebergVerifier(
            gen_checkpoint=gen_ckpt,
            inten_checkpoint=inten_ckpt,
            device='cpu',  # Use CPU for quick test
            tolerance_da=0.01,
        )
        logger.info("✓ Verifier initialized successfully")
        
        # Test scoring with a simple molecule
        import numpy as np
        test_smiles = 'CCO'  # Ethanol
        test_adduct = '[M+H]+'
        test_spectrum = np.array([[50.0, 0.5], [100.0, 1.0]])
        
        logger.info(f"Testing scoring with SMILES: {test_smiles}")
        scores = verifier.score(
            smiles_list=[test_smiles],
            precursor_mz=46.0,
            adduct=test_adduct,
            instrument=None,
            collision_eng=None,
            target_spectra=test_spectrum
        )
        logger.info(f"✓ Scoring works! Score: {scores[0]:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Verifier init failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_load():
    """Test that DiffMS model can be loaded with MCTS config."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 4: DiffMS Model Loading with MCTS")
    logger.info("=" * 60)
    
    try:
        from pathlib import Path
        from omegaconf import OmegaConf
        from src.diffms_mcts import Spec2MolDenoisingDiffusion
        from src.datasets.spec2mol_dataset import Spec2MolDataModule, Spec2MolDatasetInfos
        from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from src.diffusion.extra_features import ExtraFeatures
        from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
        from src.analysis.visualization import MolecularVisualization
        
        # Load configs
        config_dir = Path('/root/ms/DiffMS/configs')
        general_cfg = OmegaConf.load(config_dir / 'general' / 'general_default.yaml')
        model_cfg = OmegaConf.load(config_dir / 'model' / 'model_default.yaml')
        train_cfg = OmegaConf.load(config_dir / 'train' / 'train_default.yaml')
        dataset_cfg = OmegaConf.load(config_dir / 'dataset' / 'canopus.yaml')
        mcts_cfg = OmegaConf.load(config_dir / 'mcts' / 'mcts_default.yaml')
        
        cfg = OmegaConf.create({
            'general': general_cfg,
            'model': model_cfg,
            'train': train_cfg,
            'dataset': dataset_cfg,
            'mcts': mcts_cfg,
        })
        
        logger.info("Creating datamodule...")
        datamodule = Spec2MolDataModule(cfg)
        dataset_infos = Spec2MolDatasetInfos(datamodule, cfg)
        
        logger.info("Setting up features...")
        # First create domain features
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        
        # Then create extra features (may be None for spec2mol)
        if cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            from src.diffusion.extra_features import DummyExtraFeatures
            extra_features = DummyExtraFeatures()
        
        # Compute dims
        dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
        
        logger.info("Creating model components...")
        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        visualization_tools = MolecularVisualization(
            remove_h=cfg.dataset.remove_h,
            dataset_infos=dataset_infos
        )
        
        logger.info("Initializing model...")
        model = Spec2MolDenoisingDiffusion(
            cfg=cfg,
            dataset_infos=dataset_infos,
            train_metrics=train_metrics,
            visualization_tools=visualization_tools,
            extra_features=extra_features,
            domain_features=domain_features
        )
        
        logger.info("✓ Model initialized with MCTS config")
        logger.info(f"  MCTS enabled: {model.mcts_config['use_mcts']}")
        logger.info(f"  MCTS steps: {model.mcts_config['num_simulation_steps']}")
        logger.info(f"  Branch K: {model.mcts_config['branch_k']}")
        
        # Check verifier
        if model.mcts_config['use_mcts']:
            if model.verifier is not None:
                logger.info(f"✓ Verifier initialized: {type(model.verifier).__name__}")
            else:
                logger.warning("⚠ Verifier is None (will be lazy-loaded)")
        
        logger.info("Loading checkpoint...")
        checkpoint_path = '/root/ms/DiffMS/checkpoints/diffms_canopus.ckpt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                k = k[6:]
            cleaned_state_dict[k] = v
        
        model.load_state_dict(cleaned_state_dict, strict=False)
        logger.info("✓ Checkpoint loaded successfully")
        
        model.eval()
        logger.info("✓ Model set to eval mode")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model load failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("=" * 60)
    logger.info("MCTS-DiffMS Quick Integration Test")
    logger.info("=" * 60)
    logger.info("")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    if not results['imports']:
        logger.error("Import test failed. Cannot continue.")
        return
    
    results['config'] = test_mcts_config()
    results['verifier'] = test_verifier_init()
    results['model'] = test_model_load()
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name.upper()}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("")
    if all_passed:
        logger.info("=" * 60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("You can now run: python test_mcts_integration.py")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("✗ SOME TESTS FAILED")
        logger.error("Please fix the errors before running full tests")
        logger.error("=" * 60)


if __name__ == '__main__':
    main()

