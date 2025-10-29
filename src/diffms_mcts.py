import os
import time
import logging
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.mcts_utils import extract_from_dataset_batch
from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from src.metrics.train_metrics import TrainLossDiscrete
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL, CrossEntropyMetric
from src.metrics.diffms_metrics import K_ACC_Collection, K_SimilarityCollection, Validity
from src import utils
from src.mist.models.spectra_encoder import SpectraEncoderGrowing
from dataclasses import dataclass
import numpy as np
from typing import Any, Optional, List, Dict, Tuple
from .mcts_verifier import build_verifier

# optional imports deferred in verifier to avoid heavy import at module load
@dataclass
class Candicate:
    X_s: torch.Tensor
    E_s: torch.Tensor
    y_s: torch.Tensor
    prior_logp: float
    extra: dict

@dataclass
class MctsNode:
    state: dict                 # {t_int, t_norm, s_int, s_norm, X_t, E_t, y_t, node_mask}
    children: list              # list of child nodes
    parent: Optional["MctsNode"]
    terminal: bool
    # NVrU semantics
    N: int                      # number of evaluations aggregated into V
    V: float                    # state value (mean of rollouts/evals)
    r: float                    # this node's own immediate evaluation result
    # best tracking
    best_smiles: Optional[str]
    best_score: float


@dataclass
class BatchedMctsTree:
    """
    Batched MCTS tree structure using PyTorch tensors for parallel operations.
    
    Inspired by mctx (https://github.com/deepmind/mctx) but implemented in PyTorch.
    All tensors have batch dimension [B, ...] to enable parallel MCTS across multiple samples.
    
    Tree structure: Each node can have up to K children (branch_k).
    Nodes are stored in a flat array [B, max_nodes], using indices to represent parent-child relationships.
    
    Constants:
        ROOT_INDEX = 0: Root node is always at index 0
        UNVISITED = -1: Marker for unvisited children
        NO_PARENT = -1: Marker for root (no parent)
    """
    # Core MCTS statistics
    # Shape: [batch_size, max_nodes]
    node_visits: torch.Tensor     # N in UCT formula: number of times node visited
    node_values: torch.Tensor     # V in UCT formula: cumulative value (sum of rewards)
    node_rewards: torch.Tensor    # r: individual reward from last evaluation
    
    # Tree topology
    # Shape: [batch_size, max_nodes]
    parents: torch.Tensor         # Parent node index for each node, NO_PARENT for root
    
    # Children structure
    # Shape: [batch_size, max_nodes, branch_k]
    children_index: torch.Tensor  # Child node indices, UNVISITED if not expanded
    children_visits: torch.Tensor # Visit counts for each child (redundant with node_visits but useful for selection)
    children_values: torch.Tensor # Values for each child (redundant but useful for UCT)
    
    # State embeddings - molecular graph representations at each node
    # Shape: [batch_size, max_nodes, n_atoms, ...]
    node_states_X: torch.Tensor   # Node features (atom types) [B, N, n_atoms, X_dim]
    node_states_E: torch.Tensor   # Edge features (bond types) [B, N, n_atoms, n_atoms, E_dim]
    node_states_y: torch.Tensor   # Global features [B, N, y_dim]
    node_masks: torch.Tensor      # Node masks [B, N, n_atoms]
    
    # Timestep and normalization info for diffusion
    # Shape: [batch_size, max_nodes]
    node_timesteps_int: torch.Tensor    # Integer timestep t_int
    node_timesteps_norm: torch.Tensor   # Normalized timestep t_norm = t_int / T
    node_s_norm: torch.Tensor           # s_norm = (t_int - 1) / T for denoising
    
    # Terminal status
    # Shape: [batch_size, max_nodes]
    is_terminal: torch.Tensor     # Boolean: whether node is terminal (t_int == 0)
    
    # Tracking per sample
    # Shape: [batch_size]
    num_nodes: torch.Tensor       # Current number of nodes allocated per sample
    best_scores: torch.Tensor     # Best score found so far per sample
    
    # Best molecules tracking (not tensorizable due to variable-length strings)
    best_smiles: List[List[str]] # [batch_size][...] list of best SMILES per sample
    
    # Constants (class variables)
    ROOT_INDEX: int = 0
    UNVISITED: int = -1
    NO_PARENT: int = -1
    
    @property
    def batch_size(self) -> int:
        return self.node_visits.shape[0]
    
    @property
    def max_nodes(self) -> int:
        return self.node_visits.shape[1]
    
    @property
    def branch_k(self) -> int:
        return self.children_index.shape[2]

class Spec2MolDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.decoder_dtype = torch.float32
        self.T = cfg.model.diffusion_steps
        self.val_num_samples = cfg.general.val_samples_to_generate
        self.test_num_samples = cfg.general.test_samples_to_generate

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_k_acc = K_ACC_Collection(list(range(1, self.val_num_samples + 1)))
        self.val_sim_metrics = K_SimilarityCollection(list(range(1, self.val_num_samples + 1)))
        self.val_validity = Validity()
        self.val_CE = CrossEntropyMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_k_acc = K_ACC_Collection(list(range(1, self.test_num_samples + 1)))
        self.test_sim_metrics = K_SimilarityCollection(list(range(1, self.test_num_samples + 1)))
        self.test_validity = Validity()
        self.test_CE = CrossEntropyMetric()

        self.train_metrics = train_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.decoder = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        try:
            if cfg.general.decoder is not None:
                state_dict = torch.load(cfg.general.decoder, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                    
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        k = k[6:]
                        cleaned_state_dict[k] = v

                self.decoder.load_state_dict(cleaned_state_dict)
        except Exception as e:
            logging.info(f"Could not load decoder: {e}")

        hidden_size = 256
        try:
            hidden_size = cfg.model.encoder_hidden_dim
        except:
            print("No hidden size specified, using default value of 256")

        magma_modulo = 512
        try:
            magma_modulo = cfg.model.encoder_magma_modulo
        except:
            print("No magma modulo specified, using default value of 512")
        
        self.encoder = SpectraEncoderGrowing(
                        inten_transform='float',
                        inten_prob=0.1,
                        remove_prob=0.5,
                        peak_attn_layers=2,
                        num_heads=8,
                        pairwise_featurization=True,
                        embed_instrument=False,
                        cls_type='ms1',
                        set_pooling='cls',
                        spec_features='peakformula',
                        mol_features='fingerprint',
                        form_embedder='pos-cos',
                        output_size=4096,
                        hidden_size=hidden_size,
                        spectra_dropout=0.1,
                        top_layers=1,
                        refine_layers=4,
                        magma_modulo=magma_modulo,
                    )
        
        try:
            if cfg.general.encoder is not None:
                self.encoder.load_state_dict(torch.load(cfg.general.encoder), strict=True)
        except Exception as e:
            logging.info(f"Could not load encoder: {e}")

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps)
        self.denoise_nodes = getattr(cfg.dataset, 'denoise_nodes', False)
        self.merge = getattr(cfg.dataset, 'merge', 'none')

        if self.merge == 'merge-encoder_output-linear':
            self.merge_function = nn.Linear(hidden_size, cfg.dataset.morgan_nbits)
        elif self.merge == 'merge-encoder_output-mlp':
            self.merge_function = nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.SiLU(),
                nn.Linear(1024, cfg.dataset.morgan_nbits)
            )
        elif self.merge == 'downproject_4096':
            self.merge_function = nn.Linear(4096, cfg.dataset.morgan_nbits)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            logging.info(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.best_val_nll = 1e8
        self.val_counter = 1
        
        # Initialize MCTS configuration
        self._init_mcts_config()

    def _init_mcts_config(self):
        # Read MCTS config with safe defaults
        mcts = getattr(self.cfg, 'mcts', None)
        def _get(name, default):
            return getattr(mcts, name, default) if mcts is not None else default
        self.mcts_config = {
            'use_mcts': _get('use_mcts', False),
            'num_simulation_steps': _get('num_simulation_steps', _get('num_sumulation_steps', 400)),
            'branch_k': _get('branch_k', 6),
            'c_puct': _get('c_puct', _get('c_uct', 1.0)),
            'time_budget_s': _get('time_budget_s', 0.0),
            'verifier_batch_size': _get('verifier_batch_size', 32),
            't_thresh': _get('t_thresh', 10),
        }
        # External verifier should be injected; we only call verifier.score()
        self.verifier = getattr(self, 'verifier', None)
        # Caches
        self._mcts_logits_cache = {}
        self._smiles_score_cache = {}
        self._verifier_ready = False

    def _ensure_verifier(self):
        if self._verifier_ready and self.verifier is not None:
            return
        self.verifier = build_verifier(self.cfg)
        self._verifier_ready = True

    def on_test_epoch_start(self) -> None:
        if self.global_rank == 0:
            logging.info("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_k_acc.reset()
        self.test_sim_metrics.reset()
        self.test_validity.reset()
        self.test_CE.reset()

    def test_step(self, batch, i):
        output, aux = self.encoder(batch)

        data = batch["graph"]
        if self.merge == 'mist_fp':
            data.y = aux["int_preds"][-1]
        if self.merge == 'merge-encoder_output-linear':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'merge-encoder_output-mlp':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'downproject_4096':
            data.y = self.merge_function(output)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask)
        pred.X = dense_data.X
        pred.Y = data.y

        nll = 0.0 

        true_E = torch.reshape(dense_data.E, (-1, dense_data.E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))   # (bs * n * n, de)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        self.test_CE(flat_pred_E, flat_true_E)

        true_mols = [Chem.inchi.MolFromInchi(data.get_example(idx).inchi) for idx in range(len(data))] # Is this correct?
        predicted_mols = [list() for _ in range(len(data))]

        if self.global_rank == 0:
            logging.info(f"Batch {i}: Generating {self.test_num_samples} molecules for {len(data)} samples...")

        env_metas, spectra_arrays = extract_from_dataset_batch(batch, self.trainer.datamodule.test_dataset)
        mcts_results = self.mcts_sample_batch(data, env_metas, spectra_arrays)
        for idx, sample_results in enumerate(mcts_results):
            # Extract molecules from top-k results
            for smi, score, mol in sample_results:
                predicted_mols[idx].append(mol) # [bs, num_predictions]
                
        with open(f"preds/{self.name}_rank_{self.global_rank}_pred_{i}.pkl", "wb") as f:
            pickle.dump(predicted_mols, f)
        with open(f"preds/{self.name}_rank_{self.global_rank}_true_{i}.pkl", "wb") as f:
            pickle.dump(true_mols, f)
        
        for idx in range(len(data)):
            self.test_k_acc.update(predicted_mols[idx], true_mols[idx])
            self.test_sim_metrics.update(predicted_mols[idx], true_mols[idx])
            self.test_validity.update(predicted_mols[idx])

        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [
            self.test_nll.compute(), 
            self.test_X_kl.compute(), 
            self.test_E_kl.compute(),
            self.test_X_logp.compute(), 
            self.test_E_logp.compute(),
            self.test_CE.compute()
        ]

        log_dict = {
            "test/NLL": metrics[0],
            "test/X_KL": metrics[1],
            "test/E_KL": metrics[2],
            "test/X_logp": metrics[3],
            "test/E_logp": metrics[4],
            "test/E_CE": metrics[5]
        }

        self.log_dict(log_dict, sync_dist=True)
        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- Test Edge type KL: {metrics[2] :.2f} -- Test Edge type logp: {metrics[3] :.2f} -- Test Edge type CE: {metrics[5] :.2f}")

        log_dict = {}
        for key, value in self.test_k_acc.compute().items():
            log_dict[f"test/{key}"] = value
        for key, value in self.test_sim_metrics.compute().items():
            log_dict[f"test/{key}"] = value
        log_dict["test/validity"] = self.test_validity.compute()

        self.log_dict(log_dict, sync_dist=True)
        
    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        lowest_t = 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = X
        if self.denoise_nodes:
            X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.decoder(X, E, y, node_mask)

    def _terminal_check_and_smiles(self, X_0: torch.Tensor, E_0: torch.Tensor) -> Tuple[bool, Optional[str], Optional[Chem.Mol]]:

        mol = self.visualization_tools.mol_from_graphs(X_0, E_0)
        try:
            smi = Chem.MolToSmiles(mol)
            return True, smi, mol
        except:
            return False, None, None

    @torch.no_grad()
    def mcts_sample_batch(self, data: Batch, env_metas: List[dict], spectra: List[np.ndarray]) -> List[List[Tuple[str, float, Chem.Mol]]]:
        """
        Run batched MCTS for entire batch simultaneously.
        
        This replaces the sequential loop over samples with fully batched operations.
        
        Algorithm:
        1. Initialize batched tree with root states
        2. Pre-diffuse from T to t_thresh (batched)
        3. For num_simulations iterations:
           - Select: pick one leaf per sample using UCT (batched)
           - Expand: create K children for each selected leaf (batched)
           - Evaluate: score all new children with batched verifier (batched)
           - Backup: propagate values to roots (batched)
        4. Extract top-k results from each tree
        
        Args:
            data: Batch of graph data from PyTorch Geometric
            env_metas: List[dict] metadata per sample
            spectra: List[np.ndarray] target spectra per sample
            
        Returns:
            List of List of (smiles, score, mol) tuples, one list per sample
        """
        # Get number of graphs in batch using PyG's batch tensor
        # data.batch is a tensor [num_total_nodes] where each value indicates which graph the node belongs to
        # The maximum value + 1 gives us the batch size
        # This matches how to_dense() and other parts of the code compute batch_size
        batch_size = int(data.batch.max().item()) + 1
        
        # Dynamically calculate max_nodes based on config
        # Upper bound: each simulation could create branch_k new nodes
        # Add buffer for safety
        max_nodes = self.mcts_config['num_simulation_steps'] * self.mcts_config['branch_k'] + 100
        
        if self.global_rank == 0:
            logging.info(f"Starting batched MCTS with batch_size={batch_size}, max_nodes={max_nodes}")
        
        # Initialize batched tree
        tree = self._initialize_batched_tree(data, max_nodes)
        
        # Pre-diffuse root states from T to t_thresh
        tree = self._batched_prediffuse(tree, self.mcts_config['t_thresh'])
        
        # Main MCTS loop: run num_simulation_steps iterations
        num_simulations = self.mcts_config['num_simulation_steps']
        K = self.mcts_config['branch_k']
        c_puct = self.mcts_config['c_puct']
        
        for sim_step in range(num_simulations):
            # 1. Selection: pick one leaf per sample using UCT
            # Output: [batch_size] leaf indices
            leaf_indices = self._batched_select(tree, c_puct)
            
            # 2. Expansion: create K children for each selected leaf
            # Output: tree with new children, [batch_size, K] new child indices
            tree, new_child_indices = self._batched_expand(tree, leaf_indices, K)
            
            # 3. Evaluation: score all new children (batched verifier call)
            # Input: [batch_size, K] node indices
            # Output: [batch_size, K] scores
            # current_time = time.time()
            child_scores = self._batched_evaluate(tree, new_child_indices, env_metas, spectra)
            # logging.info(f"Time taken for scoring: {time.time() - current_time} seconds")
            # 4. Backup: propagate scores up to root
            # Input: [batch_size, K] node indices and scores
            tree = self._batched_backup(tree, new_child_indices, child_scores)
            
            if self.global_rank == 0 and (sim_step + 1) % 50 == 0:
                logging.info(f"Completed {sim_step + 1}/{num_simulations} MCTS simulations")
        
        # Extract results from each tree
        results = self._extract_batched_results(tree, env_metas, spectra)
        
        return results
    
    @torch.no_grad()
    def _extract_batched_results(
        self,
        tree: BatchedMctsTree,
        env_metas: List[dict],
        spectra: List[np.ndarray]
    ) -> List[List[Tuple[str, float, Chem.Mol]]]:
        """
        Extract top-k results from batched MCTS trees.
        
        **KEY OPTIMIZATION**: Batch ALL molecule scoring across all samples
        into a single verifier call instead of scoring one at a time.
        
        Algorithm:
        1. Collect all terminal nodes from all samples, sorted by Q-value
        2. If we have enough terminal nodes (>= top_k), return top_k based on Q-value
        3. If not enough terminal nodes, complete top non-terminal nodes to t=0
        4. **BATCH SCORE ALL MOLECULES AT ONCE** using verifier.score_batch()
        5. Scatter scores back to samples and return top-k per sample
        
        Args:
            tree: Batched MCTS tree after search
            env_metas: Metadata per sample
            spectra: Target spectra per sample
            
        Returns:
            List of List of (smiles, score, mol) tuples
        """
        batch_size = tree.batch_size
        device = tree.node_visits.device
        
        if self.global_rank == 0:
            logging.info(f"Extracting results from MCTS trees...")
        
        # Phase 1: Collect all molecules across all samples
        # Track: (sample_idx, mol, smi, metadata)
        all_mols = []
        all_smis = []
        all_precursor_mzs = []
        all_adducts = []
        all_instruments = []
        all_collision_engs = []
        all_target_specs = []
        sample_mol_map = []  # List of (sample_idx, local_idx_in_sample)
        
        # Track per-sample results structure
        sample_seen_smiles = [set() for _ in range(batch_size)]
        sample_molecule_count = [0 for _ in range(batch_size)]
        
        # ============ VECTORIZED PHASE 1: Identify all candidate nodes across batch ============
        # Create masks for allocated nodes: [batch_size, max_nodes]
        node_idx_grid = torch.arange(tree.max_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        is_allocated = node_idx_grid < tree.num_nodes.unsqueeze(1)  # [batch_size, max_nodes]
        has_visits = tree.node_visits > 0  # [batch_size, max_nodes]
        is_valid_node = is_allocated & has_visits  # [batch_size, max_nodes]
        
        # Separate terminal and non-terminal nodes
        is_terminal_valid = is_valid_node & tree.is_terminal  # [batch_size, max_nodes]
        is_nonterminal_valid = is_valid_node & ~tree.is_terminal  # [batch_size, max_nodes]
        
        if self.global_rank == 0:
            terminal_counts = is_terminal_valid.sum(dim=1)
            nonterminal_counts = is_nonterminal_valid.sum(dim=1)
            logging.info(f"Found {terminal_counts[0].item()} terminal, {nonterminal_counts[0].item()} non-terminal nodes (sample 0)")
        
        # ============ VECTORIZED PHASE 2: Sort and select terminal nodes ============
        # Compute rewards for terminal nodes, set -inf for invalid
        terminal_rewards = torch.where(
            is_terminal_valid,
            tree.node_rewards,
            torch.tensor(float('-inf'), device=device)
        )  # [batch_size, max_nodes]
        
        # Sort terminal nodes by reward descending per sample
        sorted_rewards, sorted_indices = torch.sort(terminal_rewards, dim=1, descending=True)  # [batch_size, max_nodes]
        
        # Process terminal nodes (collect up to test_num_samples per sample)
        for b in range(batch_size):
            num_terminals = is_terminal_valid[b].sum().item()
            if self.global_rank == 0 and b == 0:
                logging.info(f"Sample {b}: Processing {min(num_terminals, self.test_num_samples)} terminal nodes")
            
            for rank in range(min(num_terminals, self.test_num_samples)):
                if sample_molecule_count[b] >= self.test_num_samples:
                    break
                
                n = sorted_indices[b, rank].item()
                reward = sorted_rewards[b, rank].item()
                
                if reward == float('-inf'):  # No more valid terminal nodes
                    break
                
                # Get final molecule state
                X_t_indices = tree.node_states_X[b, n]
                X_t = F.one_hot(X_t_indices.long(), num_classes=self.Xdim_output).float()  # [B, n_atoms, X_dim]
                E_t_indices = tree.node_states_E[b, n]
                E_t = F.one_hot(E_t_indices.long(), num_classes=self.Edim_output).float()  # [B, n_atoms, n_atoms, E_dim]
                mask_t = tree.node_masks[b, n]
                
                # Collapse one-hot to indices
                sampled_placeholder = utils.PlaceHolder(
                    X=X_t.unsqueeze(0), 
                    E=E_t.unsqueeze(0), 
                    y=tree.node_states_y[b, n].unsqueeze(0)
                )
                sampled_placeholder = sampled_placeholder.mask(mask_t.unsqueeze(0), collapse=True)
                X_collapsed = sampled_placeholder.X.squeeze(0).cpu().numpy()
                E_collapsed = sampled_placeholder.E.squeeze(0).cpu().numpy()
                
                valid, smi, mol = self._terminal_check_and_smiles(X_collapsed, E_collapsed)
                
                # Add to batch collection
                all_mols.append(mol)
                all_smis.append(smi)
                all_precursor_mzs.append(env_metas[b]['precursor_mz'])
                all_adducts.append(env_metas[b]['adduct'])
                all_instruments.append(env_metas[b]['instrument'])
                all_collision_engs.append(env_metas[b]['collision_eng'])
                all_target_specs.append(spectra[b])
                sample_mol_map.append((b, sample_molecule_count[b]))
                
                sample_seen_smiles[b].add(smi)
                sample_molecule_count[b] += 1
        
        # ============ VECTORIZED PHASE 3: Complete non-terminal nodes with DYNAMIC MASKING ============
        # Compute Q-values for non-terminal nodes
        visits_safe = torch.clamp(tree.node_visits, min=1)  # Avoid division by zero
        Q_values = tree.node_values / visits_safe  # [batch_size, max_nodes]
        nonterminal_Q = torch.where(
            is_nonterminal_valid,
            Q_values,
            torch.tensor(float('-inf'), device=device)
        )  # [batch_size, max_nodes]
        
        # Sort non-terminal nodes by Q-value descending per sample
        sorted_Q, sorted_nt_indices = torch.sort(nonterminal_Q, dim=1, descending=True)  # [batch_size, max_nodes]
        
        # Collect ALL nodes to complete (no grouping by timestep - we'll denoise simultaneously!)
        nodes_to_complete = []  # List of (batch_idx, node_idx, initial_timestep, Q_value)
        
        for b in range(batch_size):
            needed = self.test_num_samples - sample_molecule_count[b]
            if needed <= 0:
                continue
            
            num_nonterminals = is_nonterminal_valid[b].sum().item()
            if self.global_rank == 0 and b == 0:
                logging.info(f"Sample {b}: Need {needed} more molecules, selecting from {num_nonterminals} non-terminal nodes")
            
            for rank in range(min(num_nonterminals, needed)):
                n = sorted_nt_indices[b, rank].item()
                Q = sorted_Q[b, rank].item()
                
                if Q == float('-inf'):  # No more valid non-terminal nodes
                    break
                
                t_cur = tree.node_timesteps_int[b, n].item()
                nodes_to_complete.append((b, n, t_cur, Q))
        
        if self.global_rank == 0 and len(nodes_to_complete) > 0:
            max_t = max(item[2] for item in nodes_to_complete)
            logging.info(f"Completing {len(nodes_to_complete)} non-terminal nodes via SIMULTANEOUS MASKED DENOISING (max_t={max_t})")
        
        # ============ SIMULTANEOUS BATCHED DENOISING WITH DYNAMIC MASKING ============
        # Key idea: Denoise ALL nodes together, use dynamic mask to control which nodes step at each iteration
        # Each node has its own current timestep, and we decrement it as we denoise
        if len(nodes_to_complete) > 0:
            num_nodes_to_complete = len(nodes_to_complete)
            
            # Mini-batch size: tune based on GPU memory (e.g., 32, 64, 128)
            # Smaller = less memory, Larger = faster (more vectorization)
            chunk_size = self.cfg.train.eval_batch_size
            
            # Store completed states for all nodes
            completed_states = []
            
            # Process nodes in chunks
            for chunk_start in range(0, num_nodes_to_complete, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_nodes_to_complete)
                chunk = nodes_to_complete[chunk_start:chunk_end]
                chunk_size_actual = len(chunk)
                
                b_indices = [item[0] for item in chunk]
                n_indices = [item[1] for item in chunk]
                initial_timesteps = torch.tensor([item[2] for item in chunk], device=device, dtype=torch.int32)
                
                # Gather states for this chunk: [chunk_size, n_atoms, ...]
                X_all_indices = torch.stack([tree.node_states_X[b, n] for b, n in zip(b_indices, n_indices)])
                X_all = F.one_hot(X_all_indices.long(), num_classes=self.Xdim_output).float()  # [chunk_size, n_atoms, X_dim]
                E_all_indices = torch.stack([tree.node_states_E[b, n] for b, n in zip(b_indices, n_indices)])
                E_all = F.one_hot(E_all_indices.long(), num_classes=self.Edim_output).float()  # [chunk_size, n_atoms, n_atoms, E_dim]
                y_all = torch.stack([tree.node_states_y[b, n] for b, n in zip(b_indices, n_indices)])
                mask_all = torch.stack([tree.node_masks[b, n] for b, n in zip(b_indices, n_indices)])
                
                # Track current timestep for each node in chunk: [chunk_size]
                current_timesteps = initial_timesteps.clone()
                
                if self.global_rank == 0 and chunk_start == 0:
                    max_timestep = initial_timesteps.max().item()
                    logging.info(f"Processing {num_nodes_to_complete} nodes in {(num_nodes_to_complete + chunk_size - 1) // chunk_size} chunks of size {chunk_size} (max_t={max_timestep})")
                
                # Denoise iteratively: at each iteration, denoise nodes that still need denoising
                # Continue until all nodes in this chunk reach t=0
                while current_timesteps.max().item() > 0:
                    # Create active mask: which nodes still need denoising (current_timestep > 0)
                    active_mask = current_timesteps > 0  # [chunk_size]
                    
                    if not active_mask.any():
                        break
                    
                    num_active = active_mask.sum().item()
                    if self.global_rank == 0 and current_timesteps.max().item() % 20 == 0:
                        avg_t = current_timesteps[active_mask].float().mean().item()
                        logging.info(f"Chunk {chunk_start} Denoising: {num_active}/{chunk_size_actual} nodes active (avg_t={avg_t:.1f})")
                    
                    # For inactive nodes (already at t=0), set s=t=0 (no-op, but keeps batch structure)
                    s_int_per_node = torch.clamp(current_timesteps - 1, min=0)  # [chunk_size]
                    t_int_per_node = current_timesteps  # [chunk_size]
                    
                    s_arr = (s_int_per_node.float() / self.T).unsqueeze(1)  # [chunk_size, 1]
                    t_arr = (t_int_per_node.float() / self.T).unsqueeze(1)  # [chunk_size, 1]
                    
                    # Batched denoising: ALL nodes in chunk denoised together with their individual timesteps
                    sampled_s, _ = self.sample_p_zs_given_zt(s_arr, t_arr, X_all, E_all, y_all, mask_all)
                    
                    # Update ONLY active nodes using the mask
                    # For inactive nodes (already at t=0), keep their current state
                    active_mask_expanded = active_mask.view(-1, 1, 1, 1)  # [chunk_size, 1, 1, 1]
                    E_all = torch.where(active_mask_expanded, sampled_s.E, E_all)
                    
                    # Decrement timesteps for active nodes
                    current_timesteps = torch.where(active_mask, current_timesteps - 1, current_timesteps)
                
                # Store completed states from this chunk
                for idx_in_chunk in range(chunk_size_actual):
                    completed_states.append((
                        X_all[idx_in_chunk],
                        E_all[idx_in_chunk],
                        y_all[idx_in_chunk],
                        mask_all[idx_in_chunk]
                    ))
                
                # Clear chunk tensors to free memory
                del X_all, E_all, y_all, mask_all, sampled_s
                torch.cuda.empty_cache()
            
            # Process all completed molecules
            for idx, (b, n, t_init, Q) in enumerate(nodes_to_complete):
                if sample_molecule_count[b] >= self.test_num_samples:
                    continue
                
                # Get final state from completed_states
                X_final, E_final, y_final, mask_final = completed_states[idx]
                
                # Collapse to molecule
                sampled_placeholder = utils.PlaceHolder(
                    X=X_final.unsqueeze(0), 
                    E=E_final.unsqueeze(0), 
                    y=y_final.unsqueeze(0)
                )
                sampled_placeholder = sampled_placeholder.mask(mask_final.unsqueeze(0), collapse=True)
                X_collapsed = sampled_placeholder.X.squeeze(0).cpu().numpy()
                E_collapsed = sampled_placeholder.E.squeeze(0).cpu().numpy()
                
                valid, smi, mol = self._terminal_check_and_smiles(X_collapsed, E_collapsed)
                
                # Add to batch collection
                all_mols.append(mol)
                all_smis.append(smi)
                all_precursor_mzs.append(env_metas[b]['precursor_mz'])
                all_adducts.append(env_metas[b]['adduct'])
                all_instruments.append(env_metas[b]['instrument'])
                all_collision_engs.append(env_metas[b]['collision_eng'])
                all_target_specs.append(spectra[b])
                sample_mol_map.append((b, sample_molecule_count[b]))
                
                sample_seen_smiles[b].add(smi)
                sample_molecule_count[b] += 1
        
        # Phase 2: BATCH SCORE ALL MOLECULES AT ONCE
        # This is the key optimization: one verifier call instead of hundreds/thousands
        all_scores = []
        if len(all_mols) > 0:
            self._ensure_verifier()
            all_scores = self.verifier.score_batch(
                all_mols, all_smis,
                all_precursor_mzs, all_adducts,
                all_instruments, all_collision_engs,
                all_target_specs,
                bin_size=self.cfg.mcts.similarity.bin_size
            )
        
        # Phase 3: Scatter scores back to per-sample results
        # Build results structure: List[List[Tuple[str, float, mol]]]
        results = [[] for _ in range(batch_size)]
        for idx, (sample_idx, local_idx) in enumerate(sample_mol_map):
            score = float(all_scores[idx])
            results[sample_idx].append((all_smis[idx], score, all_mols[idx]))
        
        return results

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    # ==================== Batched MCTS Methods ====================
    # These methods replace the sequential MCTS with batched operations
    
    def _initialize_batched_tree(self, data: Batch, max_nodes: int) -> BatchedMctsTree:
        """
        Initialize batched MCTS tree with root nodes.
        
        Algorithm:
        - Convert dense data to batched tensors
        - Sample initial noise for edges (X stays fixed for edge-only denoising)
        - Allocate memory for max_nodes per sample
        - Set root at index 0 for all samples
        
        Args:
            data: Batch of graph data from PyTorch Geometric
            max_nodes: Maximum nodes to allocate per sample
            
        Returns:
            BatchedMctsTree with initialized roots
        """
        # Convert to dense format
        # dense_data.X: [batch_size, n_atoms, X_dim]
        # dense_data.E: [batch_size, n_atoms, n_atoms, E_dim]
        # node_mask: [batch_size, n_atoms]
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        batch_size = dense_data.X.shape[0]
        n_atoms = dense_data.X.shape[1]
        X_dim = dense_data.X.shape[2]
        E_dim = dense_data.E.shape[3]
        y_dim = data.y.shape[1]
        
        # Sample initial noise for edges (following original MCTS logic)
        # X stays as is (edge-only denoising), E is sampled from limit distribution
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X_init = dense_data.X  # [batch_size, n_atoms, X_dim]
        E_init = z_T.E         # [batch_size, n_atoms, n_atoms, E_dim]
        y_init = data.y        # [batch_size, y_dim]
        
        device = data.x.device
        branch_k = self.mcts_config['branch_k']
        
        # Allocate tensors for tree structure
        # Initialize all to zeros/defaults, will be filled as tree grows
        node_visits = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.int32)
        node_values = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.float32)
        node_rewards = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.float32)
        parents = torch.full((batch_size, max_nodes), BatchedMctsTree.NO_PARENT, device=device, dtype=torch.int32)
        
        # Children arrays: [batch_size, max_nodes, branch_k]
        children_index = torch.full((batch_size, max_nodes, branch_k), BatchedMctsTree.UNVISITED, 
                                    device=device, dtype=torch.int32)
        children_visits = torch.zeros(batch_size, max_nodes, branch_k, device=device, dtype=torch.int32)
        children_values = torch.zeros(batch_size, max_nodes, branch_k, device=device, dtype=torch.float32)
        
        # State embeddings: [batch_size, max_nodes, ...]
        node_states_X = torch.zeros(batch_size, max_nodes, n_atoms, device=device, dtype=torch.uint8)
        node_states_E = torch.zeros(batch_size, max_nodes, n_atoms, n_atoms, device=device, dtype=torch.uint8)
        node_states_y = torch.zeros(batch_size, max_nodes, y_dim, device=device, dtype=torch.float32)
        node_masks = torch.zeros(batch_size, max_nodes, n_atoms, device=device, dtype=torch.bool)
        
        # Timestep info: [batch_size, max_nodes]
        node_timesteps_int = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.int32)
        node_timesteps_norm = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.float32)
        node_s_norm = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.float32)
        
        # Terminal status: [batch_size, max_nodes]
        is_terminal = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.bool)
        
        # Tracking arrays: [batch_size]
        num_nodes = torch.ones(batch_size, device=device, dtype=torch.int32)  # Start with 1 (root)
        best_scores = torch.full((batch_size,), -1e9, device=device, dtype=torch.float32)
        best_smiles = [[] for _ in range(batch_size)]
        
        # Initialize root nodes (index 0) with starting states
       # Initialize root nodes (index 0) with starting states
        node_states_X[:, 0] = X_init.argmax(dim=-1).to(torch.uint8)
        node_states_E[:, 0] = E_init.argmax(dim=-1).to(torch.uint8)
        node_states_y[:, 0] = y_init
        node_masks[:, 0] = node_mask
        
        # Root starts at timestep T (maximum noise), will be pre-diffused to t_thresh
        # Note: self.T is the total diffusion steps (e.g., 500)
        T_max = self.T  # Start at maximum timestep
        node_timesteps_int[:, 0] = T_max
        node_timesteps_norm[:, 0] = T_max / self.T
        node_s_norm[:, 0] = (T_max - 1) / self.T
        is_terminal[:, 0] = False  # Root at T is never terminal
        
        return BatchedMctsTree(
            node_visits=node_visits,
            node_values=node_values,
            node_rewards=node_rewards,
            parents=parents,
            children_index=children_index,
            children_visits=children_visits,
            children_values=children_values,
            node_states_X=node_states_X,
            node_states_E=node_states_E,
            node_states_y=node_states_y,
            node_masks=node_masks,
            node_timesteps_int=node_timesteps_int,
            node_timesteps_norm=node_timesteps_norm,
            node_s_norm=node_s_norm,
            is_terminal=is_terminal,
            num_nodes=num_nodes,
            best_scores=best_scores,
            best_smiles=best_smiles
        )
    
    @torch.no_grad()
    def _batched_prediffuse(self, tree: BatchedMctsTree, t_thresh: int) -> BatchedMctsTree:
        """
        Pre-diffuse root states from T down to t_thresh using batched denoising.
        
        This matches the original _mcts_sample_single lines 562-571, but batched.
        
        Algorithm:
        - For each timestep from T-1 down to t_thresh, denoise in parallel
        - Update root states with denoised edges (X stays fixed)
        
        Args:
            tree: Batched tree with roots at timestep T
            t_thresh: Target timestep to diffuse down to
            
        Returns:
            Updated tree with roots at timestep t_thresh
        """
        batch_size = tree.batch_size
        device = tree.node_visits.device
        
        # Get root states: [batch_size, n_atoms, ...]
        X_cur_indices = tree.node_states_X[:, BatchedMctsTree.ROOT_INDEX]  # [B, n_atoms]
        X_cur = F.one_hot(X_cur_indices.long(), num_classes=self.Xdim_output).float()  # [B, n_atoms, X_dim]
        E_cur_indices = tree.node_states_E[:, BatchedMctsTree.ROOT_INDEX]  # [B, n_atoms, n_atoms]
        E_cur = F.one_hot(E_cur_indices.long(), num_classes=self.Edim_output).float()  # [B, n_atoms, n_atoms, E_dim]
        y_cur = tree.node_states_y[:, BatchedMctsTree.ROOT_INDEX]  # [B, y_dim]
        mask_cur = tree.node_masks[:, BatchedMctsTree.ROOT_INDEX]  # [B, n_atoms]
        
        # Diffuse from T down to t_thresh
        # Original code: for s_int in reversed(range(t_thresh, self.T))
        for s_int in reversed(range(t_thresh, self.T)):
            # s_array: [batch_size, 1]
            s_array = torch.full((batch_size, 1), s_int, dtype=torch.float32, device=device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            
            # Batched denoising: sample_p_zs_given_zt handles batch dimension
            sampled_s, _ = self.sample_p_zs_given_zt(s_norm, t_norm, X_cur, E_cur, y_cur, mask_cur)
            # Update edges only (edge-only denoising)
            E_cur = sampled_s.E
        
        # Update root states in tree
        tree.node_states_E[:, BatchedMctsTree.ROOT_INDEX] = E_cur.argmax(dim=-1).to(torch.uint8)
        tree.node_timesteps_int[:, BatchedMctsTree.ROOT_INDEX] = t_thresh
        tree.node_timesteps_norm[:, BatchedMctsTree.ROOT_INDEX] = t_thresh / self.T
        tree.node_s_norm[:, BatchedMctsTree.ROOT_INDEX] = (t_thresh - 1) / self.T
        tree.is_terminal[:, BatchedMctsTree.ROOT_INDEX] = (t_thresh == 0)
        
        return tree
    
    @torch.no_grad()
    def _batched_select(self, tree: BatchedMctsTree, c_puct: float) -> torch.Tensor:
        """
        Traverse tree from root using UCT until finding nodes ready for expansion.
        
        This implements the standard MCTS selection phase: starting from the root,
        repeatedly select the best child according to UCT until reaching a node that
        either (1) has no children yet (needs expansion) or (2) is terminal.
        
        Algorithm (vectorized across batch):
        1. Start at root nodes for all samples
        2. For each current node, compute UCT scores for all its children
        3. Select child with highest UCT score
        4. Move to that child
        5. Repeat until reaching unexpanded or terminal nodes
        
        UCT Formula for child selection:
            UCT(child) = Q(child) + c_puct * sqrt(log(N(parent)) / N(child))
            where:
            - Q(child) = V(child) / N(child): average value
            - c_puct: exploration constant
            - N: visit count
            - Unvisited children (N=0) get infinite UCT for exploration
            
        Args:
            tree: Batched MCTS tree
            c_puct: Exploration constant for UCT formula
            
        Returns:
            leaf_indices: [batch_size] indices of nodes ready for expansion
        """
        batch_size = tree.batch_size
        device = tree.node_visits.device
        branch_k = tree.branch_k
        
        # Start at root for all samples
        # Shape: [batch_size]
        current_nodes = torch.full((batch_size,), BatchedMctsTree.ROOT_INDEX, dtype=torch.int32, device=device)
        batch_range = torch.arange(batch_size, device=device)
        
        # Traverse tree until reaching unexpanded or terminal nodes
        # Maximum depth is bounded by tree size to avoid infinite loops
        max_depth = tree.max_nodes
        
        for depth in range(max_depth):
            # Check if current nodes need expansion
            # Shape: [batch_size]
            has_children = tree.children_index[batch_range, current_nodes, 0] != BatchedMctsTree.UNVISITED
            
            # Nodes ready for expansion: no children yet OR terminal nodes
            # Shape: [batch_size]
            is_terminal_node = tree.is_terminal[batch_range, current_nodes]
            ready_for_expansion = (~has_children) | is_terminal_node
            # whether current node is ok to expand or not (leaf or not)
            # has children -- not leaf
            # is terminal -- leaf
            # no children and not terminal -- not leaf
            
            # If all samples have found nodes ready for expansion, we're done
            if ready_for_expansion.all():
                break
            
            # For continuing samples: select best child using UCT
            # Get children indices for current nodes: [batch_size, branch_k]
            children_indices = tree.children_index[batch_range, current_nodes, :]  # [B, K]
            
            # Get child statistics for UCT computation using advanced indexing
            # Shape: [batch_size, branch_k]
            # Create a mask for valid children (not UNVISITED)
            valid_children_mask = (children_indices != BatchedMctsTree.UNVISITED) & (children_indices >= 0)
            
            # Initialize with zeros
            child_visits = torch.zeros(batch_size, branch_k, device=device, dtype=torch.float32)
            child_values = torch.zeros(batch_size, branch_k, device=device, dtype=torch.float32)
            
            # For valid children, gather their statistics
            # Use advanced indexing: for each (b, k), get tree.node_visits[b, children_indices[b, k]]
            # Create batch indices: [batch_size, branch_k] repeating [0, 1, 2, ..., batch_size-1]
            batch_indices_expanded = batch_range.unsqueeze(1).expand(batch_size, branch_k)
            
            # Clamp children_indices to valid range to avoid out-of-bounds (will be masked anyway)
            children_indices_clamped = torch.clamp(children_indices, min=0, max=tree.max_nodes - 1)
            
            # Gather visits and values using advanced indexing
            gathered_visits = tree.node_visits[batch_indices_expanded, children_indices_clamped].float()
            gathered_values = tree.node_values[batch_indices_expanded, children_indices_clamped].float()
            
            # Apply mask to zero out invalid children
            child_visits = torch.where(valid_children_mask, gathered_visits, child_visits)
            child_values = torch.where(valid_children_mask, gathered_values, child_values)
            
            # Compute Q-values: average value per visit
            # Shape: [batch_size, branch_k]
            epsilon = 1e-8
            Q = torch.where(child_visits > 0, child_values / child_visits, torch.zeros_like(child_values))
            
            # Get parent visit counts for exploration term
            # Shape: [batch_size]
            parent_visits = tree.node_visits[batch_range, current_nodes].float()
            
            # Compute UCT exploration term: c_puct * sqrt(log(N_parent) / N_child)
            # Shape: [batch_size, branch_k]
            exploration = c_puct * torch.sqrt(
                torch.log(parent_visits.unsqueeze(1) + epsilon) / (child_visits + epsilon)
            )
            
            # UCT score = Q + exploration
            # Shape: [batch_size, branch_k]
            uct_scores = Q + exploration
            
            # Mask out invalid children (UNVISITED)
            # Shape: [batch_size, branch_k]
            is_invalid = children_indices == BatchedMctsTree.UNVISITED
            uct_scores = torch.where(is_invalid, torch.tensor(float('inf'), device=device), uct_scores) # the inf does not matter
            
            # Check if any sample has valid children to continue with
            has_valid_children = ~is_invalid.all(dim=1)  # [batch_size]
            
            # Update should_continue: only continue if has valid children
            should_continue = ~ready_for_expansion & has_valid_children # leaf do not have valid children so it will not continue
            
            # If no samples should continue, we're done
            if not should_continue.any():
                break
            
            # Select child with highest UCT score
            # Shape: [batch_size]
            best_child_idx = torch.argmax(uct_scores, dim=1)  # Index in [0, branch_k)
            
            # Get the actual node index of the best child for each sample
            # Use advanced indexing: gather children_indices[b, best_child_idx[b]] for each b
            # Shape: [batch_size]
            next_nodes = torch.gather(children_indices, 1, best_child_idx.unsqueeze(1)).squeeze(1)
            
            # Update current_nodes only for samples that should continue
            # For samples ready for expansion, keep current node
            current_nodes = torch.where(should_continue, next_nodes, current_nodes)
        
        return current_nodes
    
    @torch.no_grad()
    def _batched_expand(self, tree: BatchedMctsTree, leaf_indices: torch.Tensor, K: int) -> Tuple[BatchedMctsTree, torch.Tensor]:
        """
        Expand K children for selected leaves in parallel across batch.
        
        Algorithm:
        - Gather states from selected leaf nodes [batch_size, ...]
        - Repeat each state K times -> [batch_size*K, ...]
        - Call sample_p_zs_given_zt once with batch_size*K samples (batched denoising)
        - Allocate K new node indices per sample
        - Store children states in tree
        - Update parent-child relationships
        
        Args:
            tree: Batched MCTS tree
            leaf_indices: [batch_size] indices of leaves to expand
            K: Number of children (branch_k from config)
            
        Returns:
            tree: Updated tree with new children
            new_child_indices: [batch_size, K] indices of newly created children
        """
        batch_size = tree.batch_size
        device = tree.node_visits.device
        batch_range = torch.arange(batch_size, device=device)
        
        # Check if leaves are terminal or already expanded
        # Skip expansion for terminal nodes or nodes that already have children
        is_terminal = tree.is_terminal[batch_range, leaf_indices]
        has_children = tree.children_index[batch_range, leaf_indices, 0] != BatchedMctsTree.UNVISITED
        can_expand = ~is_terminal & ~has_children
        
        # Allocate new node indices for children
        # Shape: [batch_size, K]
        new_child_indices = torch.zeros(batch_size, K, device=device, dtype=torch.int32)
        for b in range(batch_size):
            if can_expand[b]:
                # Allocate K consecutive nodes
                start_idx = tree.num_nodes[b]
                new_child_indices[b] = torch.arange(start_idx, start_idx + K, device=device)
                tree.num_nodes[b] += K
        
        # For terminal nodes, set new_child_indices to leaf_indices (evaluate self, don't expand)
        for b in range(batch_size):
            if is_terminal[b]:
                # Terminal nodes evaluate themselves K times (for batch structure consistency)
                new_child_indices[b] = leaf_indices[b].repeat(K)
        
        # Gather states from leaves to expand
        # Shape: [batch_size, n_atoms, ...]
        leaf_X_indices = tree.node_states_X[batch_range, leaf_indices]  # [B, n_atoms, X_dim]
        leaf_X = F.one_hot(leaf_X_indices.long(), num_classes=self.Xdim_output).float()  # [B, n_atoms, X_dim]
        leaf_E_indices = tree.node_states_E[batch_range, leaf_indices]  # [B, n_atoms, n_atoms, E_dim]
        leaf_E = F.one_hot(leaf_E_indices.long(), num_classes=self.Edim_output).float()  # [B, n_atoms, n_atoms, E_dim]
        leaf_y = tree.node_states_y[batch_range, leaf_indices]  # [B, y_dim]
        leaf_mask = tree.node_masks[batch_range, leaf_indices]  # [B, n_atoms]
        leaf_t_int = tree.node_timesteps_int[batch_range, leaf_indices]  # [B]
        
        # Repeat for K children: [batch_size*K, ...]
        # Use repeat_interleave to repeat each sample K times: [s0, s0, s1, s1, ...] not [s0, s1, s0, s1, ...]
        expanded_X = leaf_X.repeat_interleave(K, dim=0)  # [B*K, n_atoms, X_dim]
        expanded_E = leaf_E.repeat_interleave(K, dim=0)  # [B*K, n_atoms, n_atoms, E_dim]
        expanded_y = leaf_y.repeat_interleave(K, dim=0)  # [B*K, y_dim]
        expanded_mask = leaf_mask.repeat_interleave(K, dim=0)  # [B*K, n_atoms]
        expanded_t_int = leaf_t_int.repeat_interleave(K)  # [B*K]
        
        # Compute next timestep: t' = t - 1
        # Shape: [B*K]
        next_t_int = expanded_t_int - 1
        next_t_int = torch.clamp(next_t_int, min=0)  # Ensure non-negative
        
        # Prepare s_norm and t_norm for denoising
        # For child at timestep t', denoising uses s'=(t'-1)/T, t'=(t')/T
        # Shape: [B*K, 1]
        next_leaf_t_norm = (expanded_t_int.float() / self.T).unsqueeze(1)  # [B*K, 1]
        next_t_norm = (next_t_int.float() / self.T).unsqueeze(1)  # [B*K, 1]
        next_s_norm = ((next_t_int.float() - 1).clamp(min=0) / self.T).unsqueeze(1)  # [B*K, 1]
        
        # Batched denoising: sample K children per sample in one forward pass
        # sample_p_zs_given_zt handles batch dimension automatically
        # Input: [B*K, ...], Output: [B*K, ...]
        sampled_states, _ = self.sample_p_zs_given_zt(
            next_t_norm, next_leaf_t_norm, expanded_X, expanded_E, expanded_y, expanded_mask
        )
        
        # Extract sampled edges (edge-only denoising: keep X unchanged)
        # Shape: [B*K, n_atoms, n_atoms, E_dim]
        sampled_E = sampled_states.E
        
        # Reshape back to [batch_size, K, ...]
        sampled_E_reshaped = sampled_E.view(batch_size, K, *sampled_E.shape[1:])  # [B, K, n_atoms, n_atoms, E_dim]
        original_X_reshaped = expanded_X.view(batch_size, K, *expanded_X.shape[1:])  # [B, K, n_atoms, X_dim]
        next_t_int_reshaped = next_t_int.view(batch_size, K)  # [B, K]
        # Note: next_t_norm and next_s_norm are [B*K, 1], when reshaped they become [B, K] not [B, K, 1]
        next_t_norm_reshaped = next_t_norm.squeeze(1).view(batch_size, K)  # [B, K]
        next_s_norm_reshaped = next_s_norm.squeeze(1).view(batch_size, K)  # [B, K]
        # save memory for E and X
        sampled_E_indices = sampled_E_reshaped.argmax(dim=-1).to(torch.uint8)
        original_X_indices = original_X_reshaped.argmax(dim=-1).to(torch.uint8)
        
        # Store children states in tree
        for b in range(batch_size):
            if can_expand[b]:
                leaf_idx = leaf_indices[b].item()
                for k in range(K):
                    child_idx = new_child_indices[b, k].item()
                    # Store state (X stays same, E updated)
                    tree.node_states_X[b, child_idx] = original_X_indices[b, k]
                    tree.node_states_E[b, child_idx] = sampled_E_indices[b, k]
                    tree.node_states_y[b, child_idx] = leaf_y[b]
                    tree.node_masks[b, child_idx] = leaf_mask[b]
                    
                    # Store timestep info
                    tree.node_timesteps_int[b, child_idx] = next_t_int_reshaped[b, k]
                    tree.node_timesteps_norm[b, child_idx] = next_t_norm_reshaped[b, k]
                    tree.node_s_norm[b, child_idx] = next_s_norm_reshaped[b, k]
                    tree.is_terminal[b, child_idx] = (next_t_int_reshaped[b, k] == 0)
                    
                    # Set parent relationship
                    tree.parents[b, child_idx] = leaf_idx
                    
                    # Update parent's children_index
                    tree.children_index[b, leaf_idx, k] = child_idx
        
        return tree, new_child_indices
    
    @torch.no_grad()
    def _batched_evaluate(
        self,
        tree: BatchedMctsTree,
        node_indices: torch.Tensor,  # [batch_size] or [batch_size, K]
        env_metas: List[dict],
        spectra: List[np.ndarray]
    ) -> torch.Tensor:
        """
        Evaluate nodes using batched ICEBERG forward passes.
        
        Algorithm:
        - Gather states from nodes to evaluate
        - If terminal (t_int==0): use current state
        - Else: denoise one step to get X_hat, E_hat
        - Convert to molecules (RDKit, not batchable - must loop)
        - Batch call verifier.score() with ALL valid molecules at once
        - Return scores matching input shape
        
        Key optimization: Accumulate all molecules across batch, single verifier call.
        
        Args:
            tree: Batched MCTS tree
            node_indices: [batch_size] or [batch_size, K] indices to evaluate
            env_metas: List[dict] metadata per sample (length batch_size)
            spectra: List[np.ndarray] target spectra per sample (length batch_size)
            
        Returns:
            scores: Same shape as node_indices, values are similarity scores
        """
        device = tree.node_visits.device
        original_shape = node_indices.shape
        
        # Flatten to [N] for easier processing
        node_indices_flat = node_indices.flatten()  # [N] where N = batch_size or batch_size*K
        N = node_indices_flat.shape[0]
        
        # Determine which sample each node belongs to
        if len(original_shape) == 1:
            # [batch_size]: one node per sample
            sample_indices = torch.arange(N, device=device)
        else:
            # [batch_size, K]: K nodes per sample
            batch_size = original_shape[0]
            K = original_shape[1]
            sample_indices = torch.arange(batch_size, device=device).repeat_interleave(K)
        
        # Gather states from nodes to evaluate
        # batch_indices: [N] which batch element each node belongs to (for indexing tree)
        # For [batch_size] input: batch_indices = [0, 1, 2, ...batch_size-1]
        # For [batch_size, K] input: batch_indices = [0, 0, ..., 1, 1, ..., batch_size-1, batch_size-1]
        batch_indices = sample_indices
        
        # Gather node data: [N, ...]
        X_t_indices = tree.node_states_X[batch_indices, node_indices_flat]
        X_t = F.one_hot(X_t_indices.long(), num_classes=self.Xdim_output).float()  # [N, n_atoms, X_dim]
        E_t_indices = tree.node_states_E[batch_indices, node_indices_flat]
        E_t = F.one_hot(E_t_indices.long(), num_classes=self.Edim_output).float()  # [N, n_atoms, n_atoms, E_dim]
        y_t = tree.node_states_y[batch_indices, node_indices_flat]  # [N, y_dim]
        mask_t = tree.node_masks[batch_indices, node_indices_flat]  # [N, n_atoms]
        t_int = tree.node_timesteps_int[batch_indices, node_indices_flat]  # [N]
        t_norm = tree.node_timesteps_norm[batch_indices, node_indices_flat]  # [N]
        is_terminal = tree.is_terminal[batch_indices, node_indices_flat]  # [N]
        
        # Separate terminal and non-terminal nodes
        terminal_mask = is_terminal  # [N]
        
        # Initialize output X_hat, E_hat with current states
        X_hat = X_t.clone()  # [N, n_atoms, X_dim]
        E_hat = E_t.clone()  # [N, n_atoms, n_atoms, E_dim]
        
        # For non-terminal nodes: denoise one step
        non_terminal_mask = ~terminal_mask
        if non_terminal_mask.any():
            # Gather non-terminal states
            nt_X = X_t[non_terminal_mask]  # [N_nt, n_atoms, X_dim]
            nt_E = E_t[non_terminal_mask]  # [N_nt, n_atoms, n_atoms, E_dim]
            nt_y = y_t[non_terminal_mask]  # [N_nt, y_dim]
            nt_mask = mask_t[non_terminal_mask]  # [N_nt, n_atoms]
            nt_t_norm = t_norm[non_terminal_mask].unsqueeze(1)  # [N_nt, 1]
            
            # Denoise: predict X_0 and E_0
            noisy_data = {'X_t': nt_X, 'E_t': nt_E, 'y_t': nt_y, 't': nt_t_norm, 'node_mask': nt_mask}
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, nt_mask)
            
            # Sample from predicted distribution
            prob_E = F.softmax(pred.E, dim=-1)  # [N_nt, n_atoms, n_atoms, E_dim]
            prob_X = F.softmax(pred.X, dim=-1)  # [N_nt, n_atoms, X_dim]
            
            sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=nt_mask)
            X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
            E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
            
            # Edge-only denoising: use original X, denoised E
            X_hat[non_terminal_mask] = nt_X
            E_hat[non_terminal_mask] = E_s
        
        # Collapse feature dimensions to atom/bond types
        # Use PlaceHolder and mask to convert from one-hot to indices
        sampled_placeholder = utils.PlaceHolder(X=X_hat, E=E_hat, y=y_t)
        sampled_placeholder = sampled_placeholder.mask(mask_t, collapse=True)
        
        # Now X and E are collapsed: X is [N, n_atoms] with atom types, E is [N, n_atoms, n_atoms] with bond types
        X_collapsed = sampled_placeholder.X  # [N, n_atoms]
        E_collapsed = sampled_placeholder.E  # [N, n_atoms, n_atoms]
        
        mol_list = []
        smi_list = []
        valid_indices = []  # Track which nodes produced valid molecules
        
        for i in range(N):
            # Extract single molecule (now in collapsed format)
            # mol_from_graphs expects numpy arrays for proper indexing in Python loops
            X_i = X_collapsed[i].cpu().numpy()  # [n_atoms] with atom type indices
            E_i = E_collapsed[i].cpu().numpy()  # [n_atoms, n_atoms] with bond type indices
            
            # Convert to molecule using RDKit
            valid, smi, mol = self._terminal_check_and_smiles(X_i, E_i)
            mol_list.append(mol)
            smi_list.append(smi)
            valid_indices.append(i)
        
        # Initialize scores with -1.0 (invalid molecule score)
        scores = torch.full((N,), -1.0, device=device, dtype=torch.float32)
        
        # Batch evaluate all valid molecules at once
        if len(mol_list) > 0:
            # Prepare metadata for verifier
            # Each valid molecule corresponds to a sample_index
            mols_to_eval = mol_list
            smis_to_eval = smi_list
            precursor_mzs = [env_metas[sample_indices[idx].item()]['precursor_mz'] for idx in valid_indices]
            adducts = [env_metas[sample_indices[idx].item()]['adduct'] for idx in valid_indices]
            instruments = [env_metas[sample_indices[idx].item()]['instrument'] for idx in valid_indices]
            collision_engs = [env_metas[sample_indices[idx].item()]['collision_eng'] for idx in valid_indices]
            target_specs = [spectra[sample_indices[idx].item()] for idx in valid_indices]
            
            # Call verifier with all molecules at once (batched ICEBERG)
            # This is the key optimization: one verifier call instead of N calls
            self._ensure_verifier()
            batch_scores = self.verifier.score_batch(
                mols_to_eval, smis_to_eval,
                precursor_mzs, adducts, instruments, collision_engs, target_specs
            )
            
            # Convert to tensor if needed (verifier may return numpy array or list)
            if isinstance(batch_scores, torch.Tensor):
                batch_scores_tensor = batch_scores.to(device=device, dtype=torch.float32)
            elif isinstance(batch_scores, np.ndarray):
                batch_scores_tensor = torch.from_numpy(batch_scores).to(device=device, dtype=torch.float32)
            elif isinstance(batch_scores, list):
                # Convert list elements to float first to handle numpy scalars
                batch_scores_tensor = torch.tensor([float(s) for s in batch_scores], device=device, dtype=torch.float32)
            else:
                # Single scalar value - convert to 1-element tensor
                batch_scores_tensor = torch.tensor([float(batch_scores)], device=device, dtype=torch.float32)
            
            # Scatter scores back to original positions
            for i, idx in enumerate(valid_indices):
                scores[idx] = batch_scores_tensor[i]
        
        # Store scores in node_rewards BEFORE reshaping (while still flat)
        # tree.node_rewards is [batch_size, max_nodes]
        # batch_indices is [N], node_indices_flat is [N], scores is [N]
        tree.node_rewards[batch_indices, node_indices_flat] = scores.clone()
        
        # Reshape to original shape for return value
        scores = scores.view(original_shape)
        
        return scores
    
    def _batched_backup(self, tree: BatchedMctsTree, node_indices: torch.Tensor, values: torch.Tensor) -> BatchedMctsTree:
        """
        Propagate values up the tree using batched operations.
        
        Algorithm (per sample, but vectorized where possible):
        - Start from node_indices (newly evaluated nodes)
        - Update node statistics: N += 1, r = value, V = (V * (N-1) + r) / N
        - Propagate to parents iteratively until reaching roots
        - Use scatter operations for efficiency where possible
        
        MCTS backup formula:
            N(node) += 1  (visit count)
            V(node) = (V(node) * (N(node) - 1) + reward) / N(node)  (running average)
            
        Args:
            tree: Batched MCTS tree
            node_indices: [batch_size] or [batch_size, K] starting nodes for backup
            values: Same shape as node_indices, evaluation results
            
        Returns:
            Updated tree with backed-up values
        """
        device = tree.node_visits.device
        original_shape = node_indices.shape
        batch_size = tree.batch_size
        
        # Flatten for easier processing
        node_indices_flat = node_indices.flatten()  # [N]
        values_flat = values.flatten()  # [N]
        N = node_indices_flat.shape[0]
        
        # Determine which sample each node belongs to
        if len(original_shape) == 1:
            batch_indices = torch.arange(N, device=device)
        else:
            K = original_shape[1]
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(K)
        
        # Propagate values up to root for each node
        # We need to do this iteratively because different paths have different depths
        max_depth = tree.max_nodes  # Upper bound on depth
        
        current_nodes = node_indices_flat.clone()  # [N]
        current_values = values_flat.clone()  # [N]
        
        for depth in range(max_depth):
            # Update statistics for current nodes
            for i in range(N):
                b = batch_indices[i].item()
                n = current_nodes[i].item()
                
                # Skip if invalid node (reached root's parent)
                if n < 0 or n == BatchedMctsTree.NO_PARENT:
                    continue
                
                # Update visit count: N += 1
                old_visits = tree.node_visits[b, n].item()
                new_visits = old_visits + 1
                tree.node_visits[b, n] = new_visits
                
                # Update value: V = (V * (N-1) + r) / N (running average)
                reward = current_values[i].item()
                old_value = tree.node_values[b, n].item()
                new_value = old_value + reward
                tree.node_values[b, n] = new_value
                # if tree.node_rewards[b, n] > reward and depth == 1:
                #     logging.info(f"parent reward {tree.node_rewards[b, n]}, child reward {reward}")
            
            # Move to parents
            new_nodes = torch.zeros_like(current_nodes)
            for i in range(N):
                b = batch_indices[i].item()
                n = current_nodes[i].item()
                if n >= 0 and n != BatchedMctsTree.NO_PARENT:
                    parent = tree.parents[b, n].item()
                    new_nodes[i] = parent
                else:
                    new_nodes[i] = BatchedMctsTree.NO_PARENT
            
            # Check if all reached root's parent (NO_PARENT)
            if (new_nodes == BatchedMctsTree.NO_PARENT).all():
                break
            
            current_nodes = new_nodes
            # Values stay the same as we propagate up
        return tree
