import numpy as np
import torch
from rdkit import Chem
from typing import Union, Optional, List
from rdkit.Chem import Descriptors
from matchms import Spectrum
from collections import defaultdict, OrderedDict
import logging
import multiprocessing as mp
from functools import partial
import time
import os
import warnings

# Disable RDKit warnings globally
os.environ['RDKIT_CATCH_WARNINGS'] = '0'
from rdkit import RDLogger
# Disable all RDKit logging
RDLogger.DisableLog('rdApp')  # Disable application-level logging
RDLogger.DisableLog('rdMol')  # Disable molecule warnings
RDLogger.DisableLog('rdSanit')  # Disable sanitization warnings
RDLogger.DisableLog('rdGeneral')  # Disable general warnings

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*non-writable.*')

# External deps are imported inside methods to keep import-time light

# Global worker state (initialized once per worker process)
_worker_gen_model = None
_worker_inten_tp = None

# In mcts_verifier.py

# Global worker state (initialized once per worker process)
_worker_joint_model = None

def _init_worker(gen_checkpoint, inten_checkpoint, device):
    """Initialize worker with a complete JointModel."""
    global _worker_joint_model
    
    # Suppress warnings in worker processes
    import warnings
    import os
    from rdkit import RDLogger
    
    os.environ['RDKIT_CATCH_WARNINGS'] = '0'
    RDLogger.DisableLog('rdApp')  # Disable application-level logging
    RDLogger.DisableLog('rdMol')  # Disable molecule warnings
    RDLogger.DisableLog('rdSanit')  # Disable sanitization warnings
    RDLogger.DisableLog('rdGeneral')  # Disable general warnings
    warnings.filterwarnings('ignore', message='.*non-writable.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='dgl')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    import torch
    from ms_pred.dag_pred import joint_model as iceberg_joint

    # Set torch threads to 1 for CPU-bound tasks, crucial for performance
    torch.set_num_threads(1)
    
    # Load the complete joint model once
    _worker_joint_model = iceberg_joint.JointModel.from_checkpoints(
        inten_checkpoint=inten_checkpoint,
        gen_checkpoint=gen_checkpoint,
        map_location=device  # Load directly to the target device (e.g., 'cpu')
    )
    _worker_joint_model.eval()
    _worker_joint_model.to(device)

def _worker_predict_spectrum(args):
    """
    Worker function that performs the ENTIRE prediction pipeline
    and returns only the final spectrum array.
    """
    global _worker_joint_model
    mol, smi, adduct, threshold, max_nodes, device = args

    # Suppress warnings (re-applied in worker process)
    import warnings
    warnings.filterwarnings('ignore', message='.*non-writable.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='dgl')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    try:
        # predict_mol is a method of the JointModel class
        output = _worker_joint_model.predict_mol(
            mol=mol,
            smi=smi,
            adduct=adduct,
            threshold=threshold,
            device=device,
            max_nodes=max_nodes,
            binned_out=False,
        )
        
        # Aggregate fragments into a final spectrum array
        # This is the same logic you already have in _aggregate_fragments_to_spectrum
        from collections import defaultdict
        
        frags = output.get('frags', {})
        mass_to_inten = defaultdict(float)
        for _, frag_data in frags.items():
            mzs = frag_data.get("mz_charge", [])
            intens = frag_data.get("intens", [])
            for mz, inten in zip(mzs, intens):
                if inten > 0:
                    mass_to_inten[mz] += inten
        
        if not mass_to_inten:
            return np.array([[0.0, 0.0]])
            
        return np.array(sorted(mass_to_inten.items()), dtype=float)

    except Exception as e:
        import logging
        logging.error(f"Worker error on SMILES {smi}: {e}")
        return np.array([[0.0, 0.0]])

class BaseVerifier:
    """Abstract verifier interface.

    Implementations must override score() and return a List[float] similarity
    scores (higher is better) for the provided smiles_list vs the target spectra.
    """

    def score(self,
              smiles_list: Union[List[str], str],
              precursor_mz: float,
              adduct: str,
              instrument: Optional[str],
              collision_eng: Optional[float],
              target_spectra: np.ndarray) -> List[float]:
        raise NotImplementedError


class IcebergVerifier(BaseVerifier):
    def __init__(self,
                 gen_checkpoint: str,
                 inten_checkpoint: str,
                 device: Optional[str] = None,
                 tolerance_da: float = 0.01,
                 bins_upper_mz: float = 1500.0,
                 bins_count: int = 15000,
                 num_workers: int = 8):
        from ms_pred.dag_pred import joint_model as iceberg_joint
        from matchms.similarity import CosineGreedy

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # The main process no longer needs the full model, only the verifier parts
        # Create persistent worker pool with the new initializer
        self.num_workers = num_workers
        ctx = mp.get_context('spawn')
        
        # Decide the device for workers. For this CPU-heavy task, 'cpu' is best.
        worker_device = 'cpu'
        
        self.worker_pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            # Pass checkpoints and the device for workers
            initargs=(gen_checkpoint, inten_checkpoint, worker_device)
        )
        logging.info(f"Initialized persistent worker pool with {num_workers} workers on device '{worker_device}'")

        self.cosine = CosineGreedy(tolerance=tolerance_da)
        self.tolerance_da = float(tolerance_da)
        self.bins_upper_mz = float(bins_upper_mz)
        self.bins_count = int(bins_count)
        
        # # Cache FULL ICEBERG outputs: (canonical_smiles, adduct) -> output_dict
        # self.iceberg_output_cache = OrderedDict()
        # self.cache_max_size = 500
        # self.global_rank = 0
    
    def __del__(self):
        """Cleanup worker pool on deletion."""
        if hasattr(self, 'worker_pool') and self.worker_pool is not None:
            self.worker_pool.close()
            self.worker_pool.join()

    def _to_matchms(self, spectra: np.ndarray, adduct: str, precursor_mz: float = None):
        # Accept either binned vector (len==bins_count) or raw peaks (N,2)
        arr = np.asarray(spectra)
        if arr.ndim == 1 and arr.shape[0] == self.bins_count:
            mz = np.linspace(0.0, self.bins_upper_mz, self.bins_count)
            inten = arr
        elif arr.ndim == 2 and arr.shape[1] == 2:
            mz, inten = arr[:, 0], arr[:, 1]
        else:
            raise ValueError("target_spectra must be 1D binned (length bins_count) or 2D (N,2) peaks array")
        
        metadata = {'adduct': adduct}
        if precursor_mz is not None and precursor_mz > 0:
            metadata['precursor_mz'] = float(precursor_mz)
        
        # Sort by m/z (matchms requires sorted mz values)
        sort_idx = np.argsort(mz)
        mz_sorted = mz[sort_idx]
        inten_sorted = inten[sort_idx]
        
        return Spectrum(mz=mz_sorted.astype(float), intensities=inten_sorted.astype(float), metadata=metadata)
    
    def bin_spectra(self, spec: Spectrum, mz_min: int = 0, mz_max: int = 1000, bin_size: float = 10.0):
        """
        Bin spectrum to a common grid for consistent comparison.
        
        Args:
            spec: matchms Spectrum object
            mz_min: minimum m/z value (default: 0)
            mz_max: maximum m/z value (default: 1000) 
            bin_size: bin size in m/z units (default: 1.0)
            
        Returns:
            tuple: (mz_grid, binned_intensities)
        """
        # Create common m/z grid
        mz_grid = np.arange(mz_min, mz_max + bin_size, bin_size)
        binned_intensities = np.zeros(len(mz_grid))
        
        if spec is None or len(spec.peaks.mz) == 0:
            return mz_grid, binned_intensities
            
        # Get spectrum data
        mz_values = spec.peaks.mz
        intensities = spec.peaks.intensities
        
        # Bin each peak to the nearest grid point
        for mz, intensity in zip(mz_values, intensities):
            # Find the closest bin index
            bin_idx = np.round((mz - mz_min) / bin_size).astype(int)
            
            # Check if within bounds
            if bin_idx < 0:
                binned_intensities[0] += intensity
            elif bin_idx < len(mz_grid):
                # Add intensity to the bin (sum if multiple peaks fall in same bin)
                binned_intensities[bin_idx] += intensity
            else:
                binned_intensities[-1] += intensity
                
        metadata = spec.metadata
        new_spec = Spectrum(mz=mz_grid, intensities=binned_intensities, metadata=metadata)
                
        return new_spec

    @torch.no_grad()
    def score_batch(self,
                   mol_list: List[Chem.Mol],
                   smiles_list: List[str],
                   precursor_mzs: List[float],
                   adducts: List[str],
                   instruments: List[Optional[str]],
                   collision_engs: List[Optional[float]],
                   target_spectra_list: List[np.ndarray],
                   bin_size: float = 1.0) -> List[float]:
        """Batched scoring with persistent worker pool and caching."""
        if not mol_list:
            return []
        
        canonical_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mol_list]
        worker_device = 'cpu'  # Matching the device used in the initializer

        worker_args = [
            (mol, smi, adduct, 0, 100, worker_device)  # threshold=0, max_nodes=100
            for mol, smi, adduct in zip(mol_list, canonical_smiles, adducts)
        ]

        # This now returns a list of simple NumPy arrays
        current_time = time.time()
        predicted_spectra_list = self.worker_pool.map(_worker_predict_spectrum, worker_args)
        logging.info(f"Time taken for predicting spectra: {time.time() - current_time} seconds")

        scores = []
        for i, pred_spectrum in enumerate(predicted_spectra_list):
            try:
                target_spec = target_spectra_list[i]
                adduct = adducts[i]
                prec_mz = precursor_mzs[i]

                spec_t = self._to_matchms(target_spec, adduct, prec_mz)

                if pred_spectrum.shape[0] > 0 and pred_spectrum[:, 1].sum() > 0:
                    spec_p = Spectrum(
                        mz=pred_spectrum[:, 0].astype(float),
                        intensities=pred_spectrum[:, 1].astype(float),
                        metadata={'adduct': adduct, 'precursor_mz': float(prec_mz)}
                    )
                    
                    # Bin and score
                    spec_p_binned = self.bin_spectra(spec_p, bin_size=bin_size)
                    spec_t_binned = self.bin_spectra(spec_t, bin_size=bin_size)
                    
                    result = self.cosine.pair(query=spec_p_binned, reference=spec_t_binned)
                    scores.append(result['score'] if result['score'] is not None else 0.0)
                else:
                    scores.append(0.0)
            except Exception as e:
                logging.error(f"Error in scoring molecule {smiles_list[i]}: {e}")
                scores.append(0.0)
        
        return scores

def build_verifier(cfg) -> BaseVerifier:
    """Factory to build a verifier based on cfg.mcts.verifier_type.

    Supported types: 'iceberg' (default).
    Required cfg for iceberg:
      - cfg.mcts.iceberg.gen_checkpoint
      - cfg.mcts.iceberg.inten_checkpoint
      - optional: cfg.mcts.similarity.tolerance_da (float)
      - optional: cfg.mcts.bins_upper_mz, cfg.mcts.bins_count
      - optional: cfg.mcts.num_workers (int, default 8)
    """
    vt = getattr(cfg.mcts, 'verifier_type', 'iceberg')
    if vt == 'iceberg':
        mcts = cfg.mcts
        tol = getattr(mcts.similarity, 'tolerance_da', 0.01)
        upper = getattr(mcts, 'bins_upper_mz', 1500.0)
        count = getattr(mcts, 'bins_count', 15000)
        num_workers = getattr(mcts, 'num_workers', 8)  # Default 8 workers
        return IcebergVerifier(
            gen_checkpoint=mcts.iceberg.gen_checkpoint,
            inten_checkpoint=mcts.iceberg.inten_checkpoint,
            tolerance_da=tol,
            bins_upper_mz=upper,
            bins_count=count,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unsupported verifier_type: {vt}")



