import numpy as np
import torch
from rdkit import Chem
from typing import Union, Optional, List
from rdkit.Chem import Descriptors
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
        from matchms import Spectrum
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
        
        return Spectrum(mz=mz.astype(float), intensities=inten.astype(float), metadata=metadata)
    
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
        max_inten = max(mass_to_inten.values())
        if max_inten > 0:
            mass_to_inten = {mz: inten / max_inten for mz, inten in mass_to_inten.items()}
        
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
        
        from matchms import Spectrum
        scores = []
        
        for smi, mol in zip(smiles_list, mol_list):
            # try:
                # Proactively split disconnected molecules
                fragments = self._split_disconnected_mol(mol)
                # Multiple components: predict each and combine with molecular-weight weights
                specs_and_weights = []
                for frag in fragments:
                    # Skip fragments with no bonds (single atoms or invalid fragments)
                    # These would cause errors in ICEBERG's GNN processing
                    # TODO: room of speed up: filtering out the strings, instead of calling external tools to evaluate
                    if frag.GetNumBonds() == 0:
                        logging.warning(f"Skipping fragment with no bonds: {Chem.MolToSmiles(frag)}")
                        continue
                        
                    frag_smi = Chem.MolToSmiles(frag, canonical=True)
                    # Skip fragments that cannot be converted to valid SMILES
                    # if frag_smi is None or not frag_smi.strip():
                    #     logging.warning(f"Skipping fragment with invalid SMILES")
                    #     continue
                    logging.info(f"Predicting spectrum for fragment: {frag_smi}")    
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
                    result = self.cosine.pair(spec_p, spec_t)
                    s = result['score'] # float, int: cosine score and number of matched peaks
                    scores.append(s if s is not None else 0.0)
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


