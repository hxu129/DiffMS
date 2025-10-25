import numpy as np
import torch
from rdkit import Chem
from typing import Union, Optional, List
from rdkit.Chem import Descriptors
from matchms import Spectrum
from collections import defaultdict
import logging

# External deps are imported inside methods to keep import-time light


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
                 bins_count: int = 15000):
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

    def bin_spectra(self, spec: Spectrum, mz_min: int = 0, mz_max: int = 1000, bin_size: float = 5.0):
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
                   target_spectra_list: List[np.ndarray]) -> List[float]:
        """
        Batched version of score() that processes multiple molecules simultaneously.
        
        Key optimization: Batch ICEBERG forward passes for all molecules at once.
        
        Args:
            mol_list: List of RDKit molecules to evaluate
            smiles_list: List of SMILES strings (parallel to mol_list)
            precursor_mzs: List of precursor m/z values (one per molecule)
            adducts: List of adduct strings (one per molecule)
            instruments: List of instrument names (one per molecule)
            collision_engs: List of collision energies (one per molecule)
            target_spectra_list: List of target spectra arrays (one per molecule)
            
        Returns:
            List of cosine similarity scores (one per molecule)
        """
        if len(mol_list) == 0:
            return []
        
        scores = []
        
        # Process each molecule (ICEBERG batching would require model modifications)
        # For now, we batch at a higher level (in _batched_evaluate)
        # but we can still optimize by avoiding repeated tensor allocations
        for i, (mol, smi, prec_mz, adduct, instrument, coll_eng, target_spec) in enumerate(
            zip(mol_list, smiles_list, precursor_mzs, adducts, instruments, collision_engs, target_spectra_list)
        ):
            # Call original score method for each molecule
            # Note: This is a single molecule, so we pass it directly
            score_list = self.score([mol], [smi], prec_mz, adduct, instrument, coll_eng, target_spec)
            scores.append(score_list[0])
        
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
    """
    vt = getattr(cfg.mcts, 'verifier_type', 'iceberg')
    if vt == 'iceberg':
        mcts = cfg.mcts
        tol = getattr(mcts.similarity, 'tolerance_da', 0.01)
        upper = getattr(mcts, 'bins_upper_mz', 1500.0)
        count = getattr(mcts, 'bins_count', 15000)
        return IcebergVerifier(
            gen_checkpoint=mcts.iceberg.gen_checkpoint,
            inten_checkpoint=mcts.iceberg.inten_checkpoint,
            tolerance_da=tol,
            bins_upper_mz=upper,
            bins_count=count,
        )
    else:
        raise ValueError(f"Unsupported verifier_type: {vt}")



