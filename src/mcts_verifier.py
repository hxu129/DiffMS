import numpy as np
import torch
from rdkit import Chem
from typing import Union, Optional, List

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
            try:
                # Predict using old ICEBERG model (no instrument, collision_eng, precursor_mz params)
                output = self.joint_model.predict_mol(
                    mol=mol,
                    smi=smi,
                    adduct=adduct,
                    threshold=0,
                    device=self.device,
                    max_nodes=100,
                    binned_out=False,
                )
                
                # Aggregate fragments to spectrum
                frags = output.get('frags', {})
                pred_spectrum = self._aggregate_fragments_to_spectrum(frags)
                
                # Convert to matchms Spectrum
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
                    # Handle different matchms versions
                    if hasattr(result, 'score'):
                        s = result.score
                    elif hasattr(result, '__getitem__'):
                        s = result[0] if len(result) > 0 else 0.0
                    else:
                        s = float(result) if result is not None else 0.0
                    scores.append(float(s if s is not None else 0.0))
                else:
                    scores.append(0.0)
            except Exception as e:
                # Handle errors gracefully (invalid SMILES, etc.)
                print(f"Warning: Failed to score SMILES '{smi}': {e}")
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


