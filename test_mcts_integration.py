#!/usr/bin/env python3
"""
MCTS-DiffMS Integration Test Script

This script tests the MCTS-guided molecular generation using:
- Pretrained DiffMS model from checkpoints
- Old ICEBERG model as verifier/reward
- Random samples from test set

Usage:
    python test_mcts_integration.py --num_samples 10 --use_mcts
"""

import os
import sys
import logging
import argparse
import random
from pathlib import Path
import pickle
from datetime import datetime

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Add DiffMS to path (since it's installed as a library, this may not be needed)
sys.path.insert(0, '/root/ms/DiffMS')
sys.path.insert(0, '/root/ms/DiffMS/src')


def load_model_and_config(use_mcts=True):
    """Load pretrained DiffMS model with MCTS configuration."""
    logger.info("=" * 80)
    logger.info("Loading DiffMS model and configuration...")
    logger.info("=" * 80)
    
    # Import DiffMS modules
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
    
    if use_mcts:
        mcts_cfg = OmegaConf.load(config_dir / 'mcts' / 'mcts_default.yaml')
    else:
        # Empty MCTS config (disabled)
        mcts_cfg = OmegaConf.create({'use_mcts': False})
    
    # Compose config
    cfg = OmegaConf.create({
        'general': general_cfg,
        'model': model_cfg,
        'train': train_cfg,
        'dataset': dataset_cfg,
        'mcts': mcts_cfg,
    })
    
    # Set test mode
    checkpoint_path = '/root/ms/DiffMS/checkpoints/diffms_canopus.ckpt'
    cfg.general.test_only = checkpoint_path
    
    logger.info(f"MCTS enabled: {cfg.mcts.use_mcts}")
    if cfg.mcts.use_mcts:
        logger.info(f"MCTS config: {OmegaConf.to_yaml(cfg.mcts)}")
    
    # Create datamodule
    datamodule = Spec2MolDataModule(cfg)
    dataset_infos = Spec2MolDatasetInfos(datamodule, cfg)
    
    # Setup features
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    
    # Create extra features (may be None for spec2mol)
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        from src.diffusion.extra_features import DummyExtraFeatures
        extra_features = DummyExtraFeatures()
    
    dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
    
    # Create metrics and visualization
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    visualization_tools = MolecularVisualization(
        remove_h=cfg.dataset.remove_h, 
        dataset_infos=dataset_infos
    )
    
    # Create model
    logger.info(f"Loading model from {checkpoint_path}...")
    model = Spec2MolDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Clean state dict
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[6:]
        cleaned_state_dict[k] = v
    
    model.load_state_dict(cleaned_state_dict, strict=False)
    logger.info("✓ Model loaded successfully")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    logger.info(f"✓ Model on device: {device}")
    
    return model, datamodule, dataset_infos, cfg, device


def molecules_match(mol1, mol2, use_inchi=True):
    """Check if two molecules are equivalent."""
    if mol1 is None or mol2 is None:
        return False
    
    try:
        if use_inchi:
            inchi1 = Chem.inchi.MolToInchi(mol1)
            inchi2 = Chem.inchi.MolToInchi(mol2)
            return inchi1 == inchi2
        else:
            smi1 = Chem.MolToSmiles(mol1)
            smi2 = Chem.MolToSmiles(mol2)
            return smi1 == smi2
    except:
        return False


def compute_tanimoto_similarity(mol1, mol2):
    """Compute Tanimoto similarity between two molecules."""
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0


def test_single_molecule(model, batch, true_inchi, device):
    """Test generation for a single molecule.
    
    Returns:
        dict with keys: 'success', 'predictions', 'valid_count', 'similarities'
    """
    try:
        # Move batch to device
        batch_device = {}
        for k, v in batch.items():
            if hasattr(v, 'to'):
                batch_device[k] = v.to(device)
            else:
                batch_device[k] = v
        
        # Generate molecules
        with torch.no_grad():
            pred_mols, pred_smiles = model.sample_batch(
                batch_device['graph'],
                return_smiles=True
            )
        
        # Get ground truth molecule
        true_mol = Chem.inchi.MolFromInchi(true_inchi)
        
        # Evaluate predictions
        valid_count = sum(1 for m in pred_mols if m is not None)
        similarities = [compute_tanimoto_similarity(m, true_mol) for m in pred_mols]
        matches = [molecules_match(m, true_mol) for m in pred_mols]
        
        top1_match = matches[0] if len(matches) > 0 else False
        top1_similarity = similarities[0] if len(similarities) > 0 else 0.0
        max_similarity = max(similarities) if len(similarities) > 0 else 0.0
        
        return {
            'success': True,
            'predictions': pred_mols,
            'smiles': pred_smiles,
            'valid_count': valid_count,
            'similarities': similarities,
            'matches': matches,
            'top1_match': top1_match,
            'top1_similarity': top1_similarity,
            'max_similarity': max_similarity,
        }
    
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'predictions': [],
            'smiles': [],
            'valid_count': 0,
            'similarities': [],
            'matches': [],
            'top1_match': False,
            'top1_similarity': 0.0,
            'max_similarity': 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description='Test MCTS-DiffMS Integration')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples')
    parser.add_argument('--use_mcts', action='store_true',
                        help='Enable MCTS-guided generation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='mcts_test_results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("MCTS-DiffMS Integration Test")
    logger.info("=" * 80)
    logger.info(f"Mode: {'MCTS-guided' if args.use_mcts else 'Baseline'}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load model
    model, datamodule, dataset_infos, cfg, device = load_model_and_config(args.use_mcts)
    
    # Get test dataset
    datamodule.setup('test')
    test_dataset = datamodule.test_dataset
    test_size = len(test_dataset)
    logger.info(f"Test dataset size: {test_size}")
    
    # Sample random indices
    test_indices = random.sample(range(test_size), min(args.num_samples, test_size))
    logger.info(f"Testing on indices: {test_indices[:10]}{'...' if len(test_indices) > 10 else ''}")
    
    # Results storage
    results = {
        'config': {
            'use_mcts': args.use_mcts,
            'num_samples': args.num_samples,
            'seed': args.seed,
            'mcts_config': OmegaConf.to_container(cfg.mcts) if args.use_mcts else None,
        },
        'samples': [],
        'summary': {},
    }
    
    # Test each sample
    for i, idx in enumerate(test_indices):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Test {i+1}/{len(test_indices)} - Dataset Index: {idx}")
        logger.info("=" * 80)
        
        try:
            # Get sample from dataset
            sample = test_dataset[idx]
            
            # Create single-sample batch
            from torch_geometric.data import Batch as PyGBatch
            graph_batch = PyGBatch.from_data_list([sample['graph']])
            batch = {'graph': graph_batch}
            
            # Get ground truth
            true_inchi = sample['graph'].inchi
            logger.info(f"Ground truth InChI: {true_inchi[:80]}...")
            
            # Run generation
            result = test_single_molecule(model, batch, true_inchi, device)
            
            # Log results
            if result['success']:
                logger.info(f"✓ Generation successful")
                logger.info(f"  Valid predictions: {result['valid_count']}/{len(result['predictions'])}")
                logger.info(f"  Top-1 match: {'✓ YES' if result['top1_match'] else '✗ NO'}")
                logger.info(f"  Top-1 similarity: {result['top1_similarity']:.4f}")
                logger.info(f"  Max similarity: {result['max_similarity']:.4f}")
                
                if len(result['smiles']) > 0:
                    logger.info(f"  Top-1 SMILES: {result['smiles'][0]}")
            else:
                logger.info(f"✗ Generation failed: {result.get('error', 'Unknown error')}")
            
            # Store result
            results['samples'].append({
                'index': idx,
                'true_inchi': true_inchi,
                'result': result,
            })
        
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            results['samples'].append({
                'index': idx,
                'error': str(e),
            })
    
    # Compute summary statistics
    successful_samples = [s for s in results['samples'] if 'result' in s and s['result']['success']]
    
    if len(successful_samples) > 0:
        total_predictions = sum(len(s['result']['predictions']) for s in successful_samples)
        total_valid = sum(s['result']['valid_count'] for s in successful_samples)
        top1_correct = sum(1 for s in successful_samples if s['result']['top1_match'])
        
        avg_top1_sim = np.mean([s['result']['top1_similarity'] for s in successful_samples])
        avg_max_sim = np.mean([s['result']['max_similarity'] for s in successful_samples])
        
        results['summary'] = {
            'num_tested': len(test_indices),
            'num_successful': len(successful_samples),
            'total_predictions': total_predictions,
            'total_valid': total_valid,
            'validity_rate': total_valid / total_predictions if total_predictions > 0 else 0.0,
            'top1_accuracy': top1_correct / len(successful_samples),
            'avg_top1_similarity': float(avg_top1_sim),
            'avg_max_similarity': float(avg_max_sim),
        }
    else:
        results['summary'] = {
            'num_tested': len(test_indices),
            'num_successful': 0,
            'error': 'No successful samples'
        }
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for key, value in results['summary'].items():
        logger.info(f"{key}: {value}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "mcts" if args.use_mcts else "baseline"
    output_file = output_dir / f"results_{mode_str}_{timestamp}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"✓ Results saved to {output_file}")
    
    # Also save a readable text summary
    summary_file = output_dir / f"summary_{mode_str}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("MCTS-DiffMS Integration Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Mode: {'MCTS-guided' if args.use_mcts else 'Baseline'}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        f.write("Summary Statistics:\n")
        f.write("-" * 40 + "\n")
        for key, value in results['summary'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Per-Sample Results:\n")
        f.write("-" * 40 + "\n")
        for i, sample in enumerate(results['samples']):
            f.write(f"\nSample {i+1} (Index {sample['index']}):\n")
            if 'error' in sample:
                f.write(f"  Error: {sample['error']}\n")
            elif 'result' in sample and sample['result']['success']:
                res = sample['result']
                f.write(f"  Valid: {res['valid_count']}/{len(res['predictions'])}\n")
                f.write(f"  Top-1 match: {res['top1_match']}\n")
                f.write(f"  Top-1 similarity: {res['top1_similarity']:.4f}\n")
                f.write(f"  Max similarity: {res['max_similarity']:.4f}\n")
    
    logger.info(f"✓ Summary saved to {summary_file}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

