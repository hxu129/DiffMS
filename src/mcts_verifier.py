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
from hashlib import sha256

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

@torch.no_grad()
def _worker_predict_and_score_spectrum(args):
    """
    Worker function that performs the ENTIRE prediction pipeline
    and returns (score, spectrum, spec_cache_key, score_cache_key).
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
    mol, smi, t_spec, adduct, device, bin_size, spec_cache_key, score_cache_key = args

    # stage 1: predict the spectrum
    try:
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

    except Exception as e:
        logging.error(f"Worker error on SMILES {smi}: {e}")
        p_spec = np.array([[0.0, 0.0]])

    # stage 2: score the spectrum
    try:
        score = _bin_and_score_vectorized(pred_spec=p_spec, target_spec=t_spec, bin_size=bin_size)
    except Exception as e:
        logging.error(f"Worker error on scoring spectrum {smi}: {e}")
        score = 0.0
    
    return (score, p_spec, spec_cache_key, score_cache_key)

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
                 num_workers: int = 8,
                 cache_dir: str = './cache/mcts/'):
        from ms_pred.dag_pred import joint_model as iceberg_joint
        from matchms.similarity import CosineGreedy
        import os

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Define cache file paths
        self.spectra_cache_path = os.path.join(cache_dir, 'spectra_cache.pkl')
        self.scores_cache_path = os.path.join(cache_dir, 'scores_cache.pkl')
        
        # Load persistent caches from disk
        self.spectra_cache = self._load_cache(self.spectra_cache_path)
        self.scores_cache = self._load_cache(self.scores_cache_path)
        logging.info(f"Loaded persistent caches: {len(self.spectra_cache)} spectra, {len(self.scores_cache)} scores")
        
        # Initialize batch counter for periodic saves
        self.batch_counter = 0
        self.save_every_n_batches = 10
        
        # Create persistent worker pool
        ctx = mp.get_context('spawn')
        worker_device = 'cpu'  # CPU is best for this task
        
        self.worker_pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(gen_checkpoint, inten_checkpoint, worker_device)
        )
        logging.info(f"Initialized persistent worker pool with {num_workers} workers on device '{worker_device}'")

        self.cosine = CosineGreedy(tolerance=tolerance_da)
        self.tolerance_da = float(tolerance_da)
        self.bins_upper_mz = float(bins_upper_mz)
        self.bins_count = int(bins_count)
    
    def _load_cache(self, cache_path: str) -> dict:
        """Load cache from pickle file, return empty dict if not exists."""
        import pickle
        import os
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache = pickle.load(f)
                logging.info(f"Loaded cache from {cache_path}: {len(cache)} entries")
                return cache
            except Exception as e:
                logging.warning(f"Failed to load cache from {cache_path}: {e}. Starting with empty cache.")
                return {}
        else:
            logging.info(f"No cache file found at {cache_path}. Starting with empty cache.")
            return {}
    
    def _save_cache(self, cache: dict, cache_path: str):
        """Atomically save dict to pickle file using temp file + rename."""
        import pickle
        import os
        
        temp_path = cache_path + '.tmp'
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(cache, f, protocol=4)
            os.replace(temp_path, cache_path)
            logging.info(f"Saved cache to {cache_path}: {len(cache)} entries")
        except Exception as e:
            logging.error(f"Failed to save cache to {cache_path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _save_caches_to_disk(self):
        """Save both spectra and scores caches to disk."""
        self._save_cache(self.spectra_cache, self.spectra_cache_path)
        self._save_cache(self.scores_cache, self.scores_cache_path)
    
    def __del__(self):
        """Cleanup: save caches and close worker pool."""
        # Save caches before cleanup
        if hasattr(self, 'spectra_cache') and hasattr(self, 'scores_cache'):
            logging.info("Saving caches before cleanup...")
            self._save_caches_to_disk()
        
        # Close worker pool
        if hasattr(self, 'worker_pool') and self.worker_pool is not None:
            self.worker_pool.close()
            self.worker_pool.join()
    
    def get_cache_stats(self):
        """Get statistics about the persistent cache."""
        return {
            'spectra_cache_size': len(self.spectra_cache),
            'scores_cache_size': len(self.scores_cache),
            'cache_dir': self.cache_dir
        }

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
        """Batched scoring with pre-filtering and persistent caching."""
        if not mol_list:
            return []
        
        worker_device = 'cpu'
        bin_size = float(bin_size)
        
        # Phase 1: Pre-filter using cache (BEFORE multiprocessing)
        # This is the key optimization: check cache first to avoid worker pool overhead
        scores = [None] * len(mol_list)  # Initialize results
        worker_args = []  # Only uncached items
        worker_indices = []  # Track original positions
        
        cache_hits_score = 0
        cache_hits_spec = 0
        cache_misses = 0
        
        for i, (mol, smi, adduct, t_spec) in enumerate(zip(mol_list, smiles_list, adducts, target_spectra_list)):
            # Create cache keys
            spec_cache_key = (smi, adduct)
            target_hash = sha256(t_spec.tobytes()).hexdigest()
            score_cache_key = (smi, adduct, target_hash, bin_size)
            
            # Check if score is already cached
            if score_cache_key in self.scores_cache:
                scores[i] = self.scores_cache[score_cache_key]
                cache_hits_score += 1
                continue
            
            # Score not cached - need to compute it
            # Check if spectrum is cached (partial hit)
            if spec_cache_key in self.spectra_cache:
                # Spectrum cached, only need to score it
                p_spec = self.spectra_cache[spec_cache_key]
                try:
                    score = _bin_and_score_vectorized(pred_spec=p_spec, target_spec=t_spec, bin_size=bin_size)
                except Exception as e:
                    logging.error(f"Scoring error for {smi}: {e}")
                    score = 0.0
                
                scores[i] = score
                self.scores_cache[score_cache_key] = score
                cache_hits_spec += 1
                continue
            
            # Neither spectrum nor score cached - need full computation
            cache_misses += 1
            worker_args.append((mol, smi, t_spec, adduct, worker_device, bin_size, spec_cache_key, score_cache_key))
            worker_indices.append(i)
        
        logging.info(f"Cache stats: {cache_hits_score} score hits, {cache_hits_spec} spectrum hits, {cache_misses} misses (need worker pool)")
        
        # Phase 2: Compute uncached items using worker pool
        if worker_args:
            current_time = time.time()
            worker_results = self.worker_pool.map(_worker_predict_and_score_spectrum, worker_args)
            logging.info(f"Worker pool computed {len(worker_args)} items in {time.time() - current_time:.2f}s")
            
            # Phase 3: Merge results and update caches
            for idx, (score, p_spec, spec_cache_key, score_cache_key) in zip(worker_indices, worker_results):
                scores[idx] = score
                # Update both caches
                self.spectra_cache[spec_cache_key] = p_spec
                self.scores_cache[score_cache_key] = score
        
        # Phase 4: Periodic save to disk
        self.batch_counter += 1
        if self.batch_counter >= self.save_every_n_batches:
            logging.info(f"Periodic cache save triggered (batch {self.batch_counter})")
            self._save_caches_to_disk()
            self.batch_counter = 0
        
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
      - optional: cfg.mcts.cache_dir (str, default './cache/mcts/')
    """
    vt = getattr(cfg.mcts, 'verifier_type', 'iceberg')
    if vt == 'iceberg':
        mcts = cfg.mcts
        tol = getattr(mcts.similarity, 'tolerance_da', 0.01)
        upper = getattr(mcts, 'bins_upper_mz', 1500.0)
        count = getattr(mcts, 'bins_count', 15000)
        num_workers = getattr(mcts, 'num_workers', 8)  # Default 8 workers
        cache_dir = getattr(mcts, 'cache_dir', './cache/mcts/')  # Default cache directory
        return IcebergVerifier(
            gen_checkpoint=mcts.iceberg.gen_checkpoint,
            inten_checkpoint=mcts.iceberg.inten_checkpoint,
            tolerance_da=tol,
            bins_upper_mz=upper,
            bins_count=count,
            num_workers=num_workers,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unsupported verifier_type: {vt}")



