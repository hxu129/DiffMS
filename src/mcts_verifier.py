import os
os.environ['RDKIT_CATCH_WARNINGS'] = '0'
import sys
if 'rdkit' not in sys.modules:
    # RDKit not loaded yet - good, suppress before loading
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    for level in ['rdApp', 'rdApp.info', 'rdApp.warning', 'rdApp.error',
                  'rdMol', 'rdSanit', 'rdGeneral']:
        RDLogger.DisableLog(level)

import numpy as np
import torch
from rdkit import Chem
from typing import Union, Optional, List
from collections import defaultdict
from rdkit.Chem import Descriptors
from matchms import Spectrum, similarity
from collections import defaultdict, OrderedDict
import logging
import multiprocessing as mp
from functools import partial
import time
import warnings
import logging

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

# Global worker state (initialized once per worker process)
_worker_gen_model = None
_worker_inten_tp = None

# Global worker state (initialized once per worker process)
_worker_joint_model = None
cosine = similarity.CosineGreedy(tolerance=0.01)
def _bin_and_score_vectorized(pred_spec: np.ndarray, target_spec: np.ndarray, 
                                mz_min: int = 0, mz_max: int = 1000, bin_size: float = 1.0) -> float:
    """Single-pass binned cosine similarity - FASTEST version."""
    # Create grid once
    mz_grid = np.arange(mz_min, mz_max + bin_size, bin_size)
    n_bins = len(mz_grid)
    
    # Bin both spectra simultaneously
    def bin_spectrum(spec):
        mz, inten = spec[:, 0], spec[:, 1]
        bin_idx = np.clip(np.round((mz - mz_min) / bin_size).astype(int), 0, n_bins - 1)
        binned = np.bincount(bin_idx, weights=inten, minlength=n_bins)
        return binned
    
    query_binned = bin_spectrum(pred_spec)
    ref_binned = bin_spectrum(target_spec)
    
    # Fast cosine similarity
    dot = np.dot(query_binned, ref_binned)
    norm_q = np.linalg.norm(query_binned)
    norm_r = np.linalg.norm(ref_binned)
    
    if norm_q == 0 or norm_r == 0:
        return 0.0
    
    cosine_score = dot / (norm_q * norm_r)

    return cosine_score

def _init_worker(gen_checkpoint, inten_checkpoint, device):
    """Initialize worker with a complete JointModel."""
    global _worker_joint_model
    global _worker_cache

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

    # setup cache
    _worker_cache = defaultdict()

@torch.no_grad()
def _worker_predict_and_score_spectrum(args):
    """
    Worker function that performs the ENTIRE prediction pipeline
    and returns only the final spectrum array.
    """
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

    global _worker_joint_model
    global _worker_cache
    mol, t_spec, adduct, device, bin_size = args

    # stage 1: predict the spectrum
    try:
        # predict_mol is a method of the JointModel class
        smi = Chem.MolToSmiles(mol, canonical=True)
        if smi+adduct in _worker_cache:
            p_spec = _worker_cache[smi+adduct]
        else:
            output = _worker_joint_model.predict_mol(
                mol=mol,
                smi=smi,
                adduct=adduct,
                threshold=0.0,
                device=device,
                max_nodes=100,
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
            p_spec = np.array([[0.0, 0.0]])
        else:
            p_spec = np.array(list(mass_to_inten.items()), dtype=float)

        _worker_cache[smi+adduct] = p_spec

    except Exception as e:
        logging.error(f"Worker error on SMILES {smi}: {e}")
        p_spec = np.array([[0.0, 0.0]])

    # stage 2: score the spectrum
    try:
        score = _bin_and_score_vectorized(pred_spec=p_spec, target_spec=t_spec, bin_size=bin_size)
    except Exception as e:
        logging.error(f"Worker error on scoring spectrum {smi}: {e}")
        return 0.0
    
    return score

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
        
        worker_device = 'cpu'  # Matching the device used in the initializer
        bin_size = float(bin_size)

        worker_args = [
            (mol, t_spec, adduct, worker_device, bin_size)
            for mol, t_spec, adduct in zip(mol_list, target_spectra_list, adducts)
        ]

        # This now returns a list of scores
        current_time = time.time()
        scores = self.worker_pool.map(_worker_predict_and_score_spectrum, worker_args)
        logging.info(f"Time taken for predicting and scoring spectra: {time.time() - current_time} seconds")

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



