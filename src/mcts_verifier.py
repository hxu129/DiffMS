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
# External deps are imported inside methods to keep import-time light

# Global worker state (initialized once per worker process)
_worker_gen_model = None
_worker_inten_tp = None

def _init_worker(gen_checkpoint, inten_checkpoint):
    """Initialize worker process with pre-loaded models.
    
    This runs once per worker at pool creation time.
    Models are loaded into global variables and reused for all tasks.
    """
    global _worker_gen_model, _worker_inten_tp
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', message='.*non-writable.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='dgl')
    
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    import torch
    from ms_pred.dag_pred import gen_model, dag_data
    import ms_pred.common as common
    
    # Load generation model on CPU
    _worker_gen_model = gen_model.FragGNN.load_from_checkpoint(
        gen_checkpoint, map_location='cpu'
    )
    _worker_gen_model.eval()
    _worker_gen_model.to('cpu')
    
    # Create tree processor for intensity model
    # Get hyperparameters from a temporary load of inten model
    import ms_pred.dag_pred.inten_model as inten_model
    temp_inten = inten_model.IntenGNN.load_from_checkpoint(
        inten_checkpoint, map_location='cpu'
    )
    root_encode = temp_inten.root_encode
    pe_embed = temp_inten.pe_embed_k
    add_hs = temp_inten.add_hs
    del temp_inten
    
    _worker_inten_tp = dag_data.TreeProcessor(
        root_encode=root_encode, pe_embed_k=pe_embed, add_hs=add_hs
    )

def _worker_process_molecule(args):
    """Worker function that uses pre-loaded models from global state.
    
    This is called for each molecule in the batch.
    No model serialization happens here - models are already loaded.
    """
    global _worker_gen_model, _worker_inten_tp
    
    mol, smi, adduct, threshold, max_nodes = args
    
    try:
        import ms_pred.common as common
        from ms_pred.magma import fragmentation
        
        root_mol = mol
        root_smi = smi
        root_inchi = common.inchi_from_smiles(root_smi)
        
        # Generate fragmentation tree using pre-loaded model
        frag_tree_dict = _worker_gen_model.predict_mol(
            root_mol=root_mol,
            root_smi=root_smi,
            adduct=adduct,
            threshold=threshold,
            device='cpu',
            max_nodes=max_nodes,
        )
        
        frag_tree = {
            "root_inchi": root_inchi,
            "root_smi": root_smi,
            "root_mol": root_mol,
            "name": "",
            "frags": frag_tree_dict
        }
        
        # Create engine
        engine = fragmentation.FragmentEngine(
            mol_str=smi, root_mol=root_mol, mol_str_type="smiles"
        )
        
        # Process tree using pre-loaded processor
        processed = _worker_inten_tp.process_tree_inten_pred(frag_tree)
        out_tree = processed["tree"]
        
        processed_dgl = processed["dgl_tree"]
        processed_dgl["adduct"] = common.ion2onehot_pos[adduct]
        processed_dgl["name"] = ""
        
        return (engine, processed_dgl, out_tree)
    except Exception as e:
        import logging
        logging.error(f"Error processing molecule {smi}: {e}")
        return None


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
        self.joint_model = iceberg_joint.JointModel.from_checkpoints(
            inten_checkpoint=inten_checkpoint,
            gen_checkpoint=gen_checkpoint,
            map_location=self.device
        )
        self.joint_model.eval()
        self.joint_model.to(self.device)

        self.cosine = CosineGreedy(tolerance=tolerance_da)
        self.tolerance_da = float(tolerance_da)
        self.bins_upper_mz = float(bins_upper_mz)
        self.bins_count = int(bins_count)
        
        # Cache FULL ICEBERG outputs: (canonical_smiles, adduct) -> output_dict
        self.iceberg_output_cache = OrderedDict()
        self.cache_max_size = 500
        self.global_rank = 0
        
        # Create persistent worker pool with pre-loaded models
        self.num_workers = num_workers
        self.gen_checkpoint = gen_checkpoint
        self.inten_checkpoint = inten_checkpoint
        
        # Use spawn context to avoid CUDA issues
        ctx = mp.get_context('spawn')
        self.worker_pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(gen_checkpoint, inten_checkpoint)
        )
        
        logging.info(f"Initialized persistent worker pool with {num_workers} workers")
    
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
    
    def _aggregate_fragments_to_spectrum(self, frags: dict):
        """Convert ICEBERG fragment predictions into aggregated spectrum array.
        
        Args:
            frags: Dictionary of fragment predictions from ICEBERG model
            
        Returns:
            np.ndarray: (N, 2) array with [mz, intensity] columns
        """
        from collections import defaultdict
        
        mass_to_inten = defaultdict(float)
        for frag_id, frag_data in frags.items():
            mzs = frag_data.get("mz_charge", [])
            intens = frag_data.get("intens", [])
            for mz, inten in zip(mzs, intens):
                if inten > 0:
                    mass_to_inten[mz] += inten
        
        if not mass_to_inten:
            # Return empty spectrum
            return np.array([[0.0, 0.0]])
        
        # Normalize intensities
        # max_inten = max(mass_to_inten.values())
        # if max_inten > 0:
        #     mass_to_inten = {mz: inten / max_inten for mz, inten in mass_to_inten.items()}
        
        # Convert to sorted array
        spectrum = np.array(sorted(mass_to_inten.items()), dtype=float)
        return spectrum

    def _split_disconnected_mol(self, mol: Chem.Mol):
        """Split a molecule into connected components (fragments).

        Returns list of RDKit Mol fragments. If input is connected or split fails,
        returns [mol].
        
        Note: Uses sanitizeFrags=False because molecules from generative models
        may have valence errors or other issues that would cause sanitization to fail.
        """
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return [mol]
        return list(frags)

    def _predict_spectrum_for(self, mol: Chem.Mol, smi: str, adduct: str):
        """Run ICEBERG prediction and aggregate to spectrum array (N,2).

        Returns np.ndarray with columns [mz, intensity] or empty array shape (1,2)
        with zeros when prediction fails.
        """
        output = self.joint_model.predict_mol(
            mol=mol,
            smi=smi,
            adduct=adduct,
            threshold=0,
            device=self.device,
            max_nodes=100,
            binned_out=False,
        )
        frags = output.get('frags', {})
        return self._aggregate_fragments_to_spectrum(frags)

    def _combine_spectra_weighted(self, specs_and_weights: List[tuple]):
        """Combine multiple spectra using molecular-weight weights.

        Args:
            specs_and_weights: list of tuples (spectrum_array, weight)

        Returns:
            np.ndarray (M,2) combined spectrum with intensities normalized to max=1.
        """
        # Gather weighted peaks
        peaks = []
        total_w = sum(max(0.0, float(w)) for _, w in specs_and_weights)
        if total_w <= 0:
            total_w = 1.0

        for spec, w in specs_and_weights:
            if spec is None:
                continue
            arr = np.asarray(spec)
            weight = max(0.0, float(w)) / total_w
            if weight == 0.0:
                continue
            # Append weighted intensities
            for mz, inten in arr:
                peaks.append((float(mz), float(inten) * weight))

        # if not peaks:
        #     return np.array([[0.0, 0.0]])

        # Sort by m/z and merge within tolerance
        peaks.sort(key=lambda x: x[0])
        merged = []
        curr_mz, curr_int = peaks[0]
        # Keep m/z as intensity-weighted average within a cluster
        curr_weighted_mz_sum = curr_mz * curr_int

        for mz, inten in peaks[1:]:
            if abs(mz - curr_mz) <= self.tolerance_da:
                curr_weighted_mz_sum += mz * inten
                curr_int += inten
                # update representative mz as weighted average so far
                curr_mz = curr_weighted_mz_sum / max(curr_int, 1e-12)
            else:
                merged.append((curr_mz, curr_int))
                curr_mz, curr_int = mz, inten
                curr_weighted_mz_sum = mz * inten

        merged.append((curr_mz, curr_int))

        # Normalize intensities to max 1.0
        # TODO check whether we need to normalize the intensities to max 1.0
        max_int = max(i for _, i in merged)
        if max_int > 0:
            merged = [(mz, i / max_int) for mz, i in merged]

        return np.array(merged, dtype=float)

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

    def cosine_similarity_binned(self, spec1: Spectrum, spec2: Spectrum, 
                                mz_min: int = 0, mz_max: int = 1000, bin_size: float = 1.0):
        """
        Compute cosine similarity between two spectra after binning to common grid.
        
        Args:
            spec1, spec2: matchms Spectrum objects
            mz_min, mz_max, bin_size: binning parameters
            
        Returns:
            float: cosine similarity (0-1, higher is better)
        """
        # Bin both spectra to same grid
        _, intensities1 = self.bin_spectra(spec1, mz_min, mz_max, bin_size)
        _, intensities2 = self.bin_spectra(spec2, mz_min, mz_max, bin_size)
        
        # Compute cosine similarity
        dot_product = np.dot(intensities1, intensities2)
        norm1 = np.linalg.norm(intensities1)
        norm2 = np.linalg.norm(intensities2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)


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
        if len(mol_list) == 0:
            return []
        
        n_mols = len(mol_list)
        scores = [0.0] * n_mols
        
        # Canonicalize SMILES for cache lookups
        canonical_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mol_list]
        
        # Separate cached vs uncached molecules
        cache_hits = []
        cache_misses = []
        cached_outputs = {}
        
        for i, (canon_smi, adduct) in enumerate(zip(canonical_smiles, adducts)):
            cache_key = (canon_smi, adduct)
            if cache_key in self.iceberg_output_cache:
                cache_hits.append(i)
                self.iceberg_output_cache.move_to_end(cache_key)
                cached_outputs[i] = self.iceberg_output_cache[cache_key]
            else:
                cache_misses.append(i)
        
        if self.global_rank == 0 and n_mols > 0:
            logging.info(f"ICEBERG cache: {len(cache_hits)}/{n_mols} hits ({100*len(cache_hits)/max(1,n_mols):.1f}%)")
        
        # Process cache misses using persistent worker pool
        if cache_misses:
            miss_mols = [mol_list[i] for i in cache_misses]
            miss_smis = [canonical_smiles[i] for i in cache_misses]
            miss_adducts = [adducts[i] for i in cache_misses]
            
            # Step 1: Generate fragmentation trees in parallel (persistent workers!)
            worker_args = [
                (mol, smi, adduct, 0, 50)  # threshold=0, max_nodes=50
                for mol, smi, adduct in zip(miss_mols, miss_smis, miss_adducts)
            ]
            
            # Submit to persistent pool - no serialization overhead!
            current_time = time.time()
            frag_results = self.worker_pool.map(_worker_process_molecule, worker_args)
            logging.info(f"Time taken for fragmentation tree generation: {time.time() - current_time} seconds")
            # Collect valid results
            engines = []
            processed_trees = []
            out_trees = []
            valid_indices = []
            
            for idx, result in enumerate(frag_results):
                if result is not None:
                    engine, processed_dgl, out_tree = result
                    engines.append(engine)
                    processed_trees.append(processed_dgl)
                    out_trees.append(out_tree)
                    valid_indices.append(idx)
            
            if len(processed_trees) == 0:
                # All fragmentations failed
                for miss_idx in cache_misses:
                    cached_outputs[miss_idx] = {'frags': {}}
            else:
                # Step 2: Batch intensity prediction on GPU
                batch = self.joint_model.inten_collate_fn(processed_trees)
                inten_frag_ids_batch = batch["inten_frag_ids"]
                
                safe_device = lambda x: x.to(self.device) if x is not None else x
                
                frag_graphs = safe_device(batch["frag_graphs"])
                root_reprs = safe_device(batch["root_reprs"])
                ind_maps = safe_device(batch["inds"])
                num_frags = safe_device(batch["num_frags"])
                broken_bonds = safe_device(batch["broken_bonds"])
                max_remove_hs = safe_device(batch["max_remove_hs"])
                max_add_hs = safe_device(batch["max_add_hs"])
                masses = safe_device(batch["masses"])
                adducts_tensor = safe_device(batch["adducts"]).to(self.device)
                root_forms = safe_device(batch["root_form_vecs"])
                frag_forms = safe_device(batch["frag_form_vecs"])
                
                inten_preds = self.joint_model.inten_model_obj.predict(
                    graphs=frag_graphs,
                    root_reprs=root_reprs,
                    ind_maps=ind_maps,
                    num_frags=num_frags,
                    max_breaks=broken_bonds,
                    max_add_hs=max_add_hs,
                    max_remove_hs=max_remove_hs,
                    masses=masses,
                    root_forms=root_forms,
                    frag_forms=frag_forms,
                    binned_out=False,
                    adducts=adducts_tensor,
                )
                
                inten_preds_spec = inten_preds["spec"]
                
                # Step 3: Reconstruct outputs and cache them
                for valid_idx, (out_tree, engine, adduct) in enumerate(zip(out_trees, engines, [miss_adducts[i] for i in valid_indices])):
                    inten_pred = inten_preds_spec[valid_idx]
                    inten_frag_ids = inten_frag_ids_batch[valid_idx]
                    out_frags = out_tree["frags"]
                    
                    for pred, frag_id in zip(inten_pred, inten_frag_ids):
                        out_frags[frag_id]["intens"] = pred.tolist()
                        
                        import ms_pred.common as common
                        new_masses = (
                            out_frags[frag_id]["base_mass"] + engine.shift_bucket_masses
                        )
                        mz_with_charge = new_masses + common.ion2mass[adduct]
                        out_frags[frag_id]["mz_no_charge"] = new_masses.tolist()
                        out_frags[frag_id]["mz_charge"] = mz_with_charge.tolist()
                    
                    out_tree["frags"] = out_frags
                    
                    # Map back to original index
                    original_idx = cache_misses[valid_indices[valid_idx]]
                    cached_outputs[original_idx] = out_tree
                    
                    # Cache it
                    cache_key = (miss_smis[valid_indices[valid_idx]], adduct)
                    self.iceberg_output_cache[cache_key] = out_tree
                    
                    # LRU eviction
                    if len(self.iceberg_output_cache) > self.cache_max_size:
                        self.iceberg_output_cache.popitem(last=False)
        
        # Process all outputs (cached + newly computed)
        for i in range(n_mols):
            try:
                output = cached_outputs.get(i, {'frags': {}})
                smi = smiles_list[i]
                mol = mol_list[i]
                adduct = adducts[i]
                prec_mz = precursor_mzs[i]
                target_spec = target_spectra_list[i]
                
                # Aggregate fragments to spectrum
                frags = output.get('frags', {})
                frag_spectrum = self._aggregate_fragments_to_spectrum(frags)
                pred_spectrum = frag_spectrum
                
                # Convert target to matchms Spectrum
                spec_t = self._to_matchms(target_spec, adduct, prec_mz)
                
                # Convert prediction to matchms Spectrum and compute similarity
                if pred_spectrum.shape[0] > 0 and pred_spectrum[:, 1].sum() > 0:
                    pred_metadata = {'adduct': adduct}
                    if prec_mz is not None and prec_mz > 0:
                        pred_metadata['precursor_mz'] = float(prec_mz)
                    
                    spec_p = Spectrum(
                        mz=pred_spectrum[:, 0].astype(float),
                        intensities=pred_spectrum[:, 1].astype(float),
                        metadata=pred_metadata
                    )
                    
                    # Bin to common grid
                    spec_p = self.bin_spectra(spec_p, bin_size=bin_size)
                    spec_t = self.bin_spectra(spec_t, bin_size=bin_size)
                    
                    result = self.cosine.pair(query=spec_p, reference=spec_t)
                    s = result['score']
                    scores[i] = s if s is not None else 0.0
                else:
                    scores[i] = 0.0
            
            except Exception as e:
                logging.error(f"Error processing molecule {smiles_list[i]}: {e}")
                scores[i] = 0.0
        
        return scores
    
    @torch.no_grad()
    def score(self,
              mol_list: Union[List[Chem.Mol], Chem.Mol],
              smiles_list: Union[List[str], str],
              precursor_mz: float,
              adduct: str,
              instrument: Optional[str],
              collision_eng: Optional[float],
              target_spectra: np.ndarray) -> List[float]:

        if isinstance(mol_list, Chem.Mol):
            mol_list = [mol_list]
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Build target spectrum once
        spec_t = self._to_matchms(target_spectra, adduct, precursor_mz)
        
        scores = []
        
        for smi, mol in zip(smiles_list, mol_list):
            # try:
                # Proactively split disconnected molecules
                # fragments = self._split_disconnected_mol(mol)
                # Multiple components: predict each and combine with molecular-weight weights
                specs_and_weights = []
                # for frag in fragments:
                # Skip fragments with no bonds (single atoms or invalid fragments)
                # These would cause errors in ICEBERG's GNN processing
                # TODO: room of speed up: filtering out the strings, instead of calling external tools to evaluate
                # if frag.GetNumBonds() == 0:
                #     logging.warning(f"Skipping fragment with no bonds: {Chem.MolToSmiles(frag)}")
                #     continue
                        
                frag = mol
                frag_smi = Chem.MolToSmiles(frag, canonical=True)
                # Skip fragments that cannot be converted to valid SMILES
                # if frag_smi is None or not frag_smi.strip():
                #     logging.warning(f"Skipping fragment with invalid SMILES")
                #     continue
                # logging.info(f"Predicting spectrum for fragment: {frag_smi}")    
                frag_spec = self._predict_spectrum_for(frag, frag_smi, adduct)
                mw = Descriptors.MolWt(frag) if frag is not None else 0.0
                specs_and_weights.append((frag_spec, mw))
                    
                # If no valid fragments after filtering, return empty spectrum
                if not specs_and_weights:
                    pred_spectrum = np.array([[0.0, 0.0]])
                else:
                    # TODO: figure out whether this is merged or not
                    pred_spectrum = self._combine_spectra_weighted(specs_and_weights)

                # Convert to matchms Spectrum and score
                if pred_spectrum.shape[0] > 0:
                    pred_metadata = {'adduct': adduct}
                    if precursor_mz is not None and precursor_mz > 0:
                        pred_metadata['precursor_mz'] = float(precursor_mz)

                    spec_p = Spectrum(
                        mz=pred_spectrum[:, 0].astype(float),
                        intensities=pred_spectrum[:, 1].astype(float),
                        metadata=pred_metadata
                    )
                    # bin to common grid
                    spec_p = self.bin_spectra(spec_p)
                    spec_t = self.bin_spectra(spec_t)
                    result = self.cosine.pair(query=spec_p, reference=spec_t)
                    s = result['score'] # float, int: cosine score and number of matched peaks
                    scores.append(s if s is not None else 0.0)
                    # logging
                    logging.info(f"Cosine score: {s}")
                else:
                    scores.append(0.0)
            # except Exception as e:
            #     # Handle any errors gracefully (invalid molecules, prediction failures, etc.)
            #     # Return 0.0 score for molecules that cannot be processed
            #     logging.error(f"Error processing molecule {smi}: {e}")
            #     scores.append(0.0)
        
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



