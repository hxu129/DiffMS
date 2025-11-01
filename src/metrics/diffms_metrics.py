import torch
import torch.nn as nn
from torchmetrics import Metric
from collections import Counter
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from src.utils import is_valid, canonical_mol_from_inchi


class K_ACC(Metric):
    def __init__(self, k: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_inchis: List[str], true_inchi: str):
        if true_inchi in generated_inchis[: self.k]:
            self.correct += 1
        self.total += 1

    def compute(self) -> torch.Tensor:
        """Compute final top-k accuracy."""
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total.float()


class K_ACC_Collection(Metric):
    """
    A collection of K_ACC metrics for multiple values of k.
    """
    def __init__(self, k_list: List[int], dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.metrics = nn.ModuleDict()
        for k in k_list:
            self.metrics[f"acc_at_{k}"] = K_ACC(k, dist_sync_on_step=dist_sync_on_step)

    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol, scores: List[float] = None):
        # Filter out invalid molecules
        valid_mols = [(mol, i) for i, mol in enumerate(generated_mols) if is_valid(mol)]
        
        if scores is not None:
            # MCTS mode: rank by similarity scores (descending)
            # Get valid scores corresponding to valid molecules
            valid_scores = [scores[i] for _, i in valid_mols]
            valid_mol_list = [mol for mol, _ in valid_mols]
            
            # Convert to InChIs and pair with scores
            inchi_score_pairs = [(Chem.MolToInchi(mol), score) 
                                  for mol, score in zip(valid_mol_list, valid_scores)]
            
            # Deduplicate: keep highest score for each unique InChI
            inchi_best_scores = {}
            for inchi, score in inchi_score_pairs:
                if inchi not in inchi_best_scores or score > inchi_best_scores[inchi]:
                    inchi_best_scores[inchi] = score
            
            # Sort by score (descending), extract InChIs
            inchis = [inchi for inchi, _ in sorted(inchi_best_scores.items(), 
                                                     key=lambda x: x[1], reverse=True)]
        else:
            # Non-MCTS mode: rank by frequency (original behavior)
            inchis = [Chem.MolToInchi(mol) for mol, _ in valid_mols]
            inchi_counter = Counter(inchis)
            # Sort by frequency, keep unique
            inchis = [item for item, _count in inchi_counter.most_common()]
        
        true_inchi = Chem.MolToInchi(true_mol)

        # Update each K_ACC submetric
        for metric in self.metrics.values():
            metric.update(inchis, true_inchi)

    def compute(self):
        return {name: m.compute() for name, m in self.metrics.items()}

class K_TanimotoSimilarity(Metric):
    def __init__(self, k: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("similarity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
        max_sim = 0.0
        for mol in generated_mols[: self.k]:
            try:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                sim = DataStructs.TanimotoSimilarity(gen_fp, true_fp)
                max_sim = max(max_sim, sim)
            except Exception:
                pass
        self.similarity_sum += max_sim
        self.total += 1

    def compute(self) -> torch.Tensor:
        """Compute the average max Tanimoto similarity."""
        if self.total == 0:
            return torch.tensor(0.0, device=self.similarity_sum.device)
        return self.similarity_sum / self.total.float()


class K_CosineSimilarity(Metric):
    def __init__(self, k: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("similarity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
        max_sim = 0.0
        for mol in generated_mols[: self.k]:
            try:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                sim = DataStructs.CosineSimilarity(gen_fp, true_fp)
                max_sim = max(max_sim, sim)
            except Exception:
                pass
        self.similarity_sum += max_sim
        self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.similarity_sum.device)
        return self.similarity_sum / self.total.float()


class K_SimilarityCollection(Metric):
    def __init__(self, k_list: List[int], dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.metrics = nn.ModuleDict()
        for k in k_list:
            self.metrics[f"tanimoto_at_{k}"] = K_TanimotoSimilarity(k, dist_sync_on_step=dist_sync_on_step)
            self.metrics[f"cosine_at_{k}"] = K_CosineSimilarity(k, dist_sync_on_step=dist_sync_on_step)

    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol, scores: List[float] = None):
        # Filter out invalid molecules
        valid_mols = [(mol, i) for i, mol in enumerate(generated_mols) if is_valid(mol)]
        
        if scores is not None:
            # MCTS mode: rank by similarity scores (descending)
            # Get valid scores corresponding to valid molecules
            valid_scores = [scores[i] for _, i in valid_mols]
            valid_mol_list = [mol for mol, _ in valid_mols]
            
            # Convert to InChIs and pair with scores
            inchi_score_pairs = [(Chem.MolToInchi(mol), score, mol) 
                                  for mol, score in zip(valid_mol_list, valid_scores)]
            
            # Deduplicate: keep highest score for each unique InChI
            inchi_best_scores = {}
            for inchi, score, mol in inchi_score_pairs:
                if inchi not in inchi_best_scores or score > inchi_best_scores[inchi][0]:
                    inchi_best_scores[inchi] = (score, mol)
            
            # Sort by score (descending), extract InChIs
            sorted_items = sorted(inchi_best_scores.items(), key=lambda x: x[1][0], reverse=True)
            inchis = [inchi for inchi, _ in sorted_items]
        else:
            # Non-MCTS mode: rank by frequency (original behavior)
            inchis = [Chem.MolToInchi(mol) for mol, _ in valid_mols]
            inchi_counter = Counter(inchis)
            inchis = [item for item, _count in inchi_counter.most_common()]

        processed_mols = [canonical_mol_from_inchi(inchi) for inchi in inchis]

        for metric in self.metrics.values():
            metric.update(processed_mols, true_mol)

    def compute(self):
        return {name: m.compute() for name, m in self.metrics.items()}


class Validity(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("valid", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_mols: List[Chem.Mol]):
        for mol in generated_mols:
            if is_valid(mol):
                self.valid += 1
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.valid.device)
        return self.valid.float() / self.total.float()
