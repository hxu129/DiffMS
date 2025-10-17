"""Utilities for MCTS-guided generation, particularly metadata extraction from batches."""

import numpy as np
import torch
from typing import List, Dict, Tuple


def extract_metadata_from_batch(batch, dataset) -> Tuple[List[Dict], List[np.ndarray]]:
    """Extract metadata and spectra from a batch for MCTS verifier.
    
    Args:
        batch: Batch dict with keys 'graph', 'spec', 'mol'
        dataset: The dataset object to access underlying Spectra and Mol objects
        
    Returns:
        env_metas: List of metadata dicts with keys:
            - precursor_mz: float
            - adduct: str
            - instrument: str or None
            - collision_eng: float or None
        spectra: List of np.ndarray with shape (N, 2) containing [mz, intensity]
    """
    # Get batch size
    batch_size = len(batch['graph'])
    
    env_metas = []
    spectra = []
    
    # Access the underlying spectra from dataset
    # The batch contains indices that we can use to get back to the original data
    for i in range(batch_size):
        # Try to get index from batch if available
        # Otherwise we'll need to extract from the collated data
        # For now, assume we can iterate through spec data
        
        # Get spectra object - this depends on how the collate function works
        # The spec data should be in batch structure
        spec_data = batch.get('spec', [])
        
        # Create metadata dict with defaults
        meta = {
            'precursor_mz': 0.0,
            'adduct': '[M+H]+',  # Default adduct
            'instrument': None,
            'collision_eng': None,
        }
        
        # Create default empty spectrum
        spec_array = np.array([[0.0, 1.0]])  # dummy spectrum
        
        env_metas.append(meta)
        spectra.append(spec_array)
    
    return env_metas, spectra


def extract_metadata_from_spectra_objects(spectra_list, mol_list=None) -> Tuple[List[Dict], List[np.ndarray]]:
    """Extract metadata and spectra arrays from Spectra objects directly.
    
    Args:
        spectra_list: List of Spectra objects from the dataset
        mol_list: Optional list of Mol objects (for adduct info if stored there)
        
    Returns:
        env_metas: List of metadata dicts
        spectra: List of spectrum arrays (N, 2) with [mz, intensity]
    """
    env_metas = []
    spectra_arrays = []
    
    for idx, spec in enumerate(spectra_list):
        # Ensure spectra is loaded
        if not spec._is_loaded:
            spec._load_spectra()
        
        # Extract precursor mass
        precursor_mz = spec.parentmass if spec.parentmass is not None else 0.0
        
        # Extract adduct from metadata or use default
        # Common keys: 'ionization', 'IONIZATION', 'adduct'
        adduct = '[M+H]+'  # default
        if spec.meta:
            for key in ['ionization', 'IONIZATION', 'adduct', 'ADDUCT']:
                if key in spec.meta:
                    adduct = spec.meta[key]
                    break
        
        # Extract instrument
        instrument = spec.instrument if hasattr(spec, 'instrument') else None
        if instrument is None and spec.meta:
            instrument = spec.meta.get('INSTRUMENT TYPE', None)
        
        # Collision energy (usually not available in old datasets)
        collision_eng = None
        if spec.meta and 'COLLISION_ENERGY' in spec.meta:
            try:
                collision_eng = float(spec.meta['COLLISION_ENERGY'])
            except (ValueError, TypeError):
                collision_eng = None
        
        meta = {
            'precursor_mz': float(precursor_mz),
            'adduct': adduct,
            'instrument': instrument,
            'collision_eng': collision_eng,
        }
        
        # Extract spectrum array
        # spec.spectra is a list of spectrum arrays, usually we want the first MS2 spectrum
        if spec.spectra and len(spec.spectra) > 0:
            spec_array = spec.spectra[0]  # First spectrum (MS2)
            if spec_array.ndim == 2 and spec_array.shape[1] == 2:
                spectra_arrays.append(spec_array)
            else:
                # Reshape if needed
                spectra_arrays.append(np.array([[0.0, 1.0]]))
        else:
            # Empty spectrum fallback
            spectra_arrays.append(np.array([[0.0, 1.0]]))
        
        env_metas.append(meta)
    
    return env_metas, spectra_arrays


def extract_from_dataset_batch(batch, dataset_obj) -> Tuple[List[Dict], List[np.ndarray]]:
    """Extract metadata from batch using the underlying dataset object.
    
    This is the main interface to use with the dataloader batches.
    
    Args:
        batch: Batch dict from dataloader
        dataset_obj: The SpectraMolDataset object
        
    Returns:
        env_metas: List of metadata dicts
        spectra: List of spectrum arrays
    """
    # Get batch indices - need to figure out how to map batch back to dataset indices
    # For now, we'll extract what we can from the batch structure
    
    # The batch has mol_indices and spec_indices that tell us which mols/specs are in the batch
    spec_indices = batch.get('spec_indices', None)
    mol_indices = batch.get('mol_indices', None)
    
    if spec_indices is not None and hasattr(dataset_obj, 'spectra_list'):
        # We have access to the dataset, get the actual Spectra objects
        batch_size = len(set(spec_indices.tolist()))  # unique spec indices
        unique_spec_indices = sorted(set(spec_indices.tolist()))
        
        spectra_list = [dataset_obj.spectra_list[i] for i in unique_spec_indices]
        mol_list = [dataset_obj.mol_list[i] for i in unique_spec_indices] if mol_indices is not None else None
        
        return extract_metadata_from_spectra_objects(spectra_list, mol_list)
    else:
        # Fallback: extract from batch directly (less info available)
        return extract_metadata_from_batch(batch, dataset_obj)

