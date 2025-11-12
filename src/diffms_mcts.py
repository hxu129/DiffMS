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
import threading
from pathlib import Path

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
        self.eval_full_mol = cfg.mcts.eval_full_mol

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
        
        # Score@k metric: average score of top-k predictions
        self.val_score_at_k = {k: [] for k in range(1, self.val_num_samples + 1)}

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_k_acc = K_ACC_Collection(list(range(1, self.test_num_samples + 1)))
        self.test_sim_metrics = K_SimilarityCollection(list(range(1, self.test_num_samples + 1)))
        self.test_validity = Validity()
        self.test_CE = CrossEntropyMetric()
        
        # Score@k metric: average score of top-k predictions
        self.test_score_at_k = {k: [] for k in range(1, self.test_num_samples + 1)}

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
        
        # Progress monitoring for smart barrier
        self.progress_dir = Path.cwd() / 'rank_progress'
        self.progress_dir.mkdir(exist_ok=True)
        self.progress_file = self.progress_dir / f'rank_{self.global_rank}_progress.txt'
        self.last_progress_update = time.time()
        self.current_step = None
        self.current_batch = None
        
        # Initialize MCTS configuration
        self._init_mcts_config()

    def _init_mcts_config(self):
        # Read MCTS config with safe defaults
        mcts = getattr(self.cfg, 'mcts', None)
        def _get(name, default):
            return getattr(mcts, name, default) if mcts is not None else default
        
        branch_k = _get('branch_k', 6)
        use_temperature = _get('use_temperature', False)
        temperature_values = _get('temperature_values', None)
        
        # Process temperature values
        if not use_temperature:
            # If temperature is disabled, use 1.0 for all branches
            temperature_values = [1.0] * branch_k
        else:
            # Temperature is enabled - must be specified
            if temperature_values is None:
                raise ValueError(
                    f"use_temperature=True but temperature_values not specified. "
                    f"Please provide a list of {branch_k} temperature values."
                )
            # Convert to list if needed
            if not isinstance(temperature_values, list):
                temperature_values = list(temperature_values)
            # Validate length
            if len(temperature_values) != branch_k:
                raise ValueError(
                    f"Temperature values length ({len(temperature_values)}) must match branch_k ({branch_k})"
                )
        
        self.mcts_config = {
            'use_mcts': _get('use_mcts', False),
            'num_simulation_steps': _get('num_simulation_steps', _get('num_sumulation_steps', 400)),
            'branch_k': branch_k,
            'c_puct': _get('c_puct', _get('c_uct', 1.0)),
            'time_budget_s': _get('time_budget_s', 0.0),
            'verifier_batch_size': _get('verifier_batch_size', 32),
            'expand_steps': _get('expand_steps', 1),  # Number of denoising steps during expansion
            'prediffuse_steps': _get('prediffuse_steps', 10),
            'debug_logging': _get('debug_logging', False),  # Enable detailed debug logging
            # Temperature sampling for diversity
            'use_temperature': use_temperature,
            'temperature_values': temperature_values,  # List of K temperature values
        }
        # External verifier should be injected; we only call verifier.score()
        self.verifier = getattr(self, 'verifier', None)
        # Caches
        self._mcts_logits_cache = {}
        self._smiles_score_cache = {}
        self._verifier_ready = False
        
        # Debug tracking (only initialized if debug_logging is enabled)
        if self.mcts_config['debug_logging']:
            self._init_debug_tracking()

    def _init_debug_tracking(self):
        """Initialize data structures for detailed MCTS debugging."""
        self.debug_metrics = {
            'tree_size_history': [],           # Track total nodes over simulation steps
            'root_children_visits': [],         # Track root's children visit counts over time
            'reward_history': [],               # All rewards seen during search
            'q_value_history': [],              # Top nodes' Q-values over time
            'simulation_step_markers': [],      # Which simulation step each metric was recorded
        }
        # Track search paths: node_id -> (parent_id, smiles, score, discovery_step)
        self.node_provenance = {}
        logging.info("[DEBUG] MCTS debug tracking initialized")
    
    def _ensure_verifier(self):
        if self._verifier_ready and self.verifier is not None:
            return
        self.verifier = build_verifier(self.cfg, ddp_rank=self.global_rank)
        self._verifier_ready = True
    
    def _update_progress(self, status: str, batch_idx: int, progress_pct: float = 0.0):
        """
        Update progress file to indicate this rank is still working.
        
        Args:
            status: Current status (e.g., "test_step", "mcts_sampling", "completed")
            batch_idx: Current batch index
            progress_pct: Progress percentage (0-100)
        """
        try:
            current_time = time.time()
            self.last_progress_update = current_time
            self.current_step = status
            self.current_batch = batch_idx
            
            # Write progress to file (atomic write)
            progress_data = {
                'rank': self.global_rank,
                'status': status,
                'batch_idx': batch_idx,
                'progress_pct': progress_pct,
                'timestamp': current_time,
                'elapsed_time': current_time - (getattr(self, 'step_start_time', current_time))
            }
            
            # Use temp file + rename for atomic write
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                f.write(f"{progress_data['rank']}\t{progress_data['status']}\t{progress_data['batch_idx']}\t"
                       f"{progress_data['progress_pct']:.2f}\t{progress_data['timestamp']:.3f}\t"
                       f"{progress_data['elapsed_time']:.2f}\n")
            temp_file.replace(self.progress_file)
        except Exception as e:
            # Don't fail if progress update fails
            logging.debug(f"[Rank {self.global_rank}] Failed to update progress: {e}")
    
    def _check_other_ranks_progress(self, world_size: int, max_stale_time: float = 300.0) -> Dict[int, Dict]:
        """
        Check progress of other ranks to see if they're still working.
        
        Args:
            world_size: Total number of ranks
            max_stale_time: Maximum time (seconds) before considering a rank stale/dead
            
        Returns:
            Dict mapping rank -> progress info, only includes ranks that are still active
        """
        active_ranks = {}
        current_time = time.time()
        
        for rank in range(world_size):
            if rank == self.global_rank:
                continue
                
            progress_file = self.progress_dir / f'rank_{rank}_progress.txt'
            if not progress_file.exists():
                continue
            
            try:
                with open(progress_file, 'r') as f:
                    line = f.readline().strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        rank_id = int(parts[0])
                        status = parts[1]
                        batch_idx = int(parts[2])
                        progress_pct = float(parts[3])
                        timestamp = float(parts[4])
                        
                        time_since_update = current_time - timestamp
                        
                        # Only consider ranks that have updated recently
                        if time_since_update < max_stale_time:
                            active_ranks[rank_id] = {
                                'status': status,
                                'batch_idx': batch_idx,
                                'progress_pct': progress_pct,
                                'last_update': timestamp,
                                'time_since_update': time_since_update
                            }
            except Exception as e:
                logging.debug(f"[Rank {self.global_rank}] Failed to read progress for rank {rank}: {e}")
        
        return active_ranks
    
    def _smart_barrier(self, context: str = "barrier", 
                      check_interval: float = 60.0,
                      max_stale_time: float = 300.0,
                      base_timeout: float = 3600.0):
        """
        Smart barrier that waits for other ranks, but extends timeout if they're still working.
        
        Strategy:
        - If other ranks are still updating progress, keep waiting (they're working)
        - Only timeout if ranks haven't updated for max_stale_time (they're stuck/dead)
        - Dynamically adjust timeout based on observed progress
        
        Args:
            context: Context string for logging
            check_interval: How often to check other ranks' progress (seconds)
            max_stale_time: Maximum time without progress update before considering rank dead (seconds)
            base_timeout: Base timeout for barrier (seconds)
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        
        world_size = torch.distributed.get_world_size()
        start_time = time.time()
        last_check_time = start_time
        
        logging.info(f"[Rank {self.global_rank}] Entering smart barrier ({context})...")
        
        # Update our progress to indicate we're waiting at barrier
        self._update_progress(f"barrier_{context}", -1, 0.0)
        
        while True:
            elapsed = time.time() - start_time
            
            # Check other ranks' progress periodically
            if time.time() - last_check_time >= check_interval:
                active_ranks = self._check_other_ranks_progress(world_size, max_stale_time)
                last_check_time = time.time()
                
                waiting_ranks = [r for r in range(world_size) if r != self.global_rank and r not in active_ranks]
                
                # Filter active_ranks to exclude those waiting at barrier
                working_ranks = {r: info for r, info in active_ranks.items() 
                                if not info['status'].startswith('barrier_')}
                
                if len(working_ranks) > 0:
                    # Some ranks are still working - log and continue waiting
                    working_status = ", ".join([f"R{r}:{info['status']}" for r, info in working_ranks.items()])
                    logging.info(f"[Rank {self.global_rank}] Smart barrier: {len(working_ranks)} ranks still working "
                               f"({working_status}), waiting... (elapsed: {elapsed:.1f}s)")
                    # Reset timeout since we detected active ranks
                    start_time = time.time()  # Reset timeout counter
                elif len(waiting_ranks) == world_size - 1 or len(active_ranks) == world_size - 1:
                    # All other ranks are also at barrier - try actual barrier
                    try:
                        torch.distributed.barrier()
                        logging.info(f"[Rank {self.global_rank}] Smart barrier passed - all ranks synchronized")
                        break
                    except Exception as e:
                        logging.warning(f"[Rank {self.global_rank}] Barrier failed, retrying: {e}")
                        time.sleep(1)
                        continue
                else:
                    # Some ranks haven't updated in a while - check if they're truly dead
                    stale_ranks = waiting_ranks
                    logging.warning(f"[Rank {self.global_rank}] Smart barrier: {len(stale_ranks)} ranks appear stale "
                                  f"(R{stale_ranks}), but continuing to wait... (elapsed: {elapsed:.1f}s)")
            
            # Check if we've exceeded base timeout (but only if no ranks are active)
            if elapsed > base_timeout:
                active_ranks = self._check_other_ranks_progress(world_size, max_stale_time)
                if len(active_ranks) == 0:
                    # No active ranks and timeout exceeded - this might be a real problem
                    logging.error(f"[Rank {self.global_rank}] Smart barrier timeout after {elapsed:.1f}s "
                                f"with no active ranks detected. Attempting barrier anyway...")
                    try:
                        torch.distributed.barrier()
                        logging.info(f"[Rank {self.global_rank}] Barrier passed after timeout")
                        break
                    except Exception as e:
                        logging.error(f"[Rank {self.global_rank}] Barrier failed after timeout: {e}")
                        # Continue waiting - maybe ranks will recover
                        start_time = time.time()  # Reset timeout
                else:
                    # Active ranks detected - extend timeout
                    logging.info(f"[Rank {self.global_rank}] Extending timeout due to active ranks "
                               f"(elapsed: {elapsed:.1f}s)")
                    start_time = time.time()  # Reset timeout counter
            
            # Small sleep to avoid busy waiting
            time.sleep(min(check_interval / 4, 5.0))

    def on_validation_epoch_start(self) -> None:
        if self.global_rank == 0:
            logging.info("Starting validation...")
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_k_acc.reset()
        self.val_sim_metrics.reset()
        self.val_validity.reset()
        self.val_CE.reset()
        
        # Reset score@k tracking
        self.val_score_at_k = {k: [] for k in range(1, self.val_num_samples + 1)}

    def validation_step(self, batch, i):
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

        self.val_CE(flat_pred_E, flat_true_E)

        true_mols = [Chem.inchi.MolFromInchi(data.get_example(idx).inchi) for idx in range(len(data))] # Is this correct?
        predicted_mols = [list() for _ in range(len(data))]
        predicted_scores = [list() for _ in range(len(data))]

        if self.global_rank == 0:
            logging.info(f"Batch {i}: Generating {self.test_num_samples} molecules for {len(data)} samples...")

        env_metas, spectra_arrays = extract_from_dataset_batch(batch, self.trainer.datamodule.test_dataset)
        mcts_results = self.mcts_sample_batch(data, env_metas, spectra_arrays)
        for idx, sample_results in enumerate(mcts_results):
            # Extract molecules and scores from top-k results
            for smi, score, mol in sample_results:
                predicted_mols[idx].append(mol) # [bs, num_predictions]
                predicted_scores[idx].append(score) # [bs, num_predictions]
                
        with open(f"preds/{self.name}_rank_{self.global_rank}_pred_{i}.pkl", "wb") as f:
            pickle.dump(predicted_mols, f)
        with open(f"preds/{self.name}_rank_{self.global_rank}_true_{i}.pkl", "wb") as f:
            pickle.dump(true_mols, f)
        
        for idx in range(len(data)):
            # Pass scores to metrics for MCTS-based ranking
            self.val_k_acc.update(predicted_mols[idx], true_mols[idx], scores=predicted_scores[idx])
            self.val_sim_metrics.update(predicted_mols[idx], true_mols[idx], scores=predicted_scores[idx])
            self.val_validity.update(predicted_mols[idx])
            
            # Compute score@k: average score of top-k predictions
            if len(predicted_scores[idx]) > 0:
                sorted_scores = sorted(predicted_scores[idx], reverse=True)
                for k in range(1, self.val_num_samples + 1):
                    if k <= len(sorted_scores):
                        avg_score = np.mean(sorted_scores[:k])
                        self.val_score_at_k[k].append(avg_score)

        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        logging.info(f"[Rank {self.global_rank}] Entering on_validation_epoch_end at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Synchronize all processes before merging (ensure all ranks finished writing)
        # Use smart barrier that waits for active ranks but times out for stuck ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._smart_barrier(context="validation_epoch_end", 
                              check_interval=60.0,  # Check every minute
                              max_stale_time=300.0,  # 5 minutes without update = stale
                              base_timeout=3600.0)  # 1 hour base timeout
        
        # Merge rank-specific prediction files and caches (only on rank 0)
        # if self.global_rank == 0:
        #     self._merge_distributed_predictions(stage='val')
        #     self._merge_distributed_caches()
        
        metrics = [
            self.val_nll.compute(), 
            self.val_X_kl.compute(), 
            self.val_E_kl.compute(),
            self.val_X_logp.compute(), 
            self.val_E_logp.compute(),
            self.val_CE.compute()
        ]

        log_dict = {
            "val/NLL": metrics[0],
            "val/X_KL": metrics[1],
            "val/E_KL": metrics[2],
            "val/X_logp": metrics[3],
            "val/E_logp": metrics[4],
            "val/E_CE": metrics[5]
        }

        logging.info(f"[Rank {self.global_rank}] Logging validation metrics with sync_dist=True...")
        self.log_dict(log_dict, sync_dist=True)
        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- Test Edge type KL: {metrics[2] :.2f} -- Test Edge type logp: {metrics[3] :.2f} -- Test Edge type CE: {metrics[5] :.2f}")

        log_dict = {}
        for key, value in self.val_k_acc.compute().items():
            log_dict[f"val/{key}"] = value
        for key, value in self.val_sim_metrics.compute().items():
            log_dict[f"val/{key}"] = value
        log_dict["val/validity"] = self.val_validity.compute()
        
        # Add score@k metrics
        for k in [1, 5, 10, 20, 50, 100]:
            if k <= self.val_num_samples and k in self.val_score_at_k and len(self.val_score_at_k[k]) > 0:
                log_dict[f"val/score_at_{k}"] = float(np.mean(self.val_score_at_k[k]))

        logging.info(f"[Rank {self.global_rank}] Logging final validation metrics with sync_dist=True...")
        self.log_dict(log_dict, sync_dist=True)
        logging.info(f"[Rank {self.global_rank}] Completed on_validation_epoch_end")

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
        
        # Reset score@k tracking
        self.test_score_at_k = {k: [] for k in range(1, self.test_num_samples + 1)}

    def test_step(self, batch, i):
        step_start_time = time.time()
        self.step_start_time = step_start_time
        logging.info(f"[Rank {self.global_rank}] Starting test_step {i} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update progress: starting test step
        self._update_progress("test_step_start", i, 0.0)
        
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
        predicted_scores = [list() for _ in range(len(data))]

        logging.info(f"[Rank {self.global_rank}] Batch {i}: Generating {self.test_num_samples} molecules for {len(data)} samples...")

        # Update progress: starting MCTS sampling
        self._update_progress("mcts_sampling", i, 0.0)
        
        env_metas, spectra_arrays = extract_from_dataset_batch(batch, self.trainer.datamodule.test_dataset)
        mcts_start_time = time.time()
        mcts_results = self.mcts_sample_batch(data, env_metas, spectra_arrays)
        mcts_time = time.time() - mcts_start_time
        logging.info(f"[Rank {self.global_rank}] Batch {i}: MCTS sampling completed in {mcts_time:.2f} seconds")
        
        # Update progress: MCTS completed
        self._update_progress("mcts_completed", i, 50.0)
        for idx, sample_results in enumerate(mcts_results):
            # Extract molecules and scores from top-k results
            for smi, score, mol in sample_results:
                predicted_mols[idx].append(mol) # [bs, num_predictions]
                predicted_scores[idx].append(score) # [bs, num_predictions]
                
        with open(f"preds/{self.name}_rank_{self.global_rank}_pred_{i}.pkl", "wb") as f:
            pickle.dump(predicted_mols, f)
        with open(f"preds/{self.name}_rank_{self.global_rank}_true_{i}.pkl", "wb") as f:
            pickle.dump(true_mols, f)
        
        for idx in range(len(data)):
            # Pass scores to metrics for MCTS-based ranking
            self.test_k_acc.update(predicted_mols[idx], true_mols[idx], scores=predicted_scores[idx])
            self.test_sim_metrics.update(predicted_mols[idx], true_mols[idx], scores=predicted_scores[idx])
            self.test_validity.update(predicted_mols[idx])
            
            # Compute score@k: average score of top-k predictions
            if len(predicted_scores[idx]) > 0:
                sorted_scores = sorted(predicted_scores[idx], reverse=True)
                for k in range(1, self.test_num_samples + 1):
                    if k <= len(sorted_scores):
                        avg_score = np.mean(sorted_scores[:k])
                        self.test_score_at_k[k].append(avg_score)

        step_time = time.time() - step_start_time
        logging.info(f"[Rank {self.global_rank}] Completed test_step {i} in {step_time:.2f} seconds (MCTS: {mcts_time:.2f}s)")
        
        # Update progress: test step completed
        self._update_progress("test_step_completed", i, 100.0)
        
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        logging.info(f"[Rank {self.global_rank}] Entering on_test_epoch_end at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Synchronize all processes before merging (ensure all ranks finished writing)
        # Use smart barrier that waits for active ranks but times out for stuck ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._smart_barrier(context="test_epoch_end",
                              check_interval=60.0,  # Check every minute
                              max_stale_time=300.0,  # 5 minutes without update = stale
                              base_timeout=7200.0)  # 2 hours base timeout (matching DDP timeout)
        
        # Merge rank-specific prediction files and caches (only on rank 0)
        # if self.global_rank == 0:
        #     self._merge_distributed_predictions(stage='test')
        #     self._merge_distributed_caches()
        
        logging.info(f"[Rank {self.global_rank}] Computing metrics...")
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

        logging.info(f"[Rank {self.global_rank}] Logging metrics with sync_dist=True (this may wait for slow ranks)...")
        self.log_dict(log_dict, sync_dist=True)
        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- Test Edge type KL: {metrics[2] :.2f} -- Test Edge type logp: {metrics[3] :.2f} -- Test Edge type CE: {metrics[5] :.2f}")

        log_dict = {}
        for key, value in self.test_k_acc.compute().items():
            log_dict[f"test/{key}"] = value
        for key, value in self.test_sim_metrics.compute().items():
            log_dict[f"test/{key}"] = value
        log_dict["test/validity"] = self.test_validity.compute()
        
        # Add score@k metrics
        for k in [1, 5, 10, 20, 50, 100]:
            if k <= self.test_num_samples and k in self.test_score_at_k and len(self.test_score_at_k[k]) > 0:
                log_dict[f"test/score_at_{k}"] = float(np.mean(self.test_score_at_k[k]))

        logging.info(f"[Rank {self.global_rank}] Logging final metrics with sync_dist=True...")
        self.log_dict(log_dict, sync_dist=True)
        logging.info(f"[Rank {self.global_rank}] Completed on_test_epoch_end")
        
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
        smi = Chem.MolToSmiles(mol)
        if self.eval_full_mol:
            return (False, None, None) if smi is None else (True, smi, mol)
        else:
            # Get the fragment with highest molecular weights (using atom count as proxy)
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if not fragments:
                return False, None, None
            max_weight_fragment = max(fragments, key=lambda x: x.GetNumAtoms())
            try:
                smi = Chem.MolToSmiles(max_weight_fragment)
                return True, smi, max_weight_fragment
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
        tree = self._batched_prediffuse(tree, self.mcts_config['prediffuse_steps'])
        
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
            current_time = time.time()
            child_scores = self._batched_evaluate(tree, new_child_indices, env_metas, spectra)
            logging.info(f"Time taken for scoring: {time.time() - current_time} seconds")
            # 4. Backup: propagate scores up to root
            # Input: [batch_size, K] node indices and scores
            tree = self._batched_backup(tree, new_child_indices, child_scores)
            
            # Debug logging: track metrics at regular intervals
            if self.mcts_config['debug_logging'] and (sim_step + 1) % 50 == 0:
                self._log_mcts_metrics(tree, sim_step + 1, batch_size, new_child_indices)
            
            if self.global_rank == 0 and (sim_step + 1) % 50 == 0:
                logging.info(f"Completed {sim_step + 1}/{num_simulations} MCTS simulations")
        
        # Extract results from each tree
        results = self._extract_batched_results(tree, env_metas, spectra)
        
        # Final debug summary and save metrics
        if self.mcts_config['debug_logging'] and self.global_rank == 0:
            self._log_final_debug_summary(tree, results, batch_size)
            self._save_debug_metrics()
            self._save_tree_structure(tree, sample_idx=0)  # Save tree for sample 0
        
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
                    # Use temperature=1.0 (standard sampling, no diversity needed when completing nodes)
                    sampled_s, _ = self.sample_p_zs_given_zt(s_arr, t_arr, X_all, E_all, y_all, mask_all, temperature=1.0)
                    
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

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, temperature=1.0):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well
           
           Args:
               s, t: Normalized timesteps [bs, 1]
               X_t, E_t, y_t: Current noisy states
               node_mask: Node mask [bs, n_atoms]
               temperature: Temperature for sampling diversity
                   - float: single temperature for all samples (default 1.0)
                   - torch.Tensor [bs]: per-sample temperature values
                       Used in MCTS expand to create diverse children
        """
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

        # Apply temperature scaling before softmax
        if isinstance(temperature, torch.Tensor):
            # Per-sample temperature for MCTS diversity: [bs] -> [bs, 1, 1] for X, [bs, 1, 1, 1] for E
            temp_X = temperature.view(bs, 1, 1)
            temp_E = temperature.view(bs, 1, 1, 1)
            pred_X = F.softmax(pred.X / temp_X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E / temp_E, dim=-1)  # bs, n, n, d0
        else:
            # Single temperature (usually 1.0 for standard sampling)
            pred_X = F.softmax(pred.X / temperature, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E / temperature, dim=-1)  # bs, n, n, d0

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

    # ==================== Debug Logging Methods ====================
    
    def _log_mcts_metrics(self, tree: BatchedMctsTree, sim_step: int, batch_size: int, 
                         newly_evaluated_nodes: torch.Tensor = None):
        """
        Log key MCTS metrics for debugging.
        
        Tracks:
        - Tree size (total nodes)
        - Root children visit counts (exploration balance)
        - Reward distribution (sparsity analysis) - ONLY for newly evaluated nodes in this step
        - Q-values of top nodes
        - Terminal node information
        """
        device = tree.node_visits.device
        
        # Focus on first sample for detailed logging (to avoid clutter)
        sample_idx = 2
        
        # 1. Tree size
        tree_size = tree.num_nodes[sample_idx].item()
        self.debug_metrics['tree_size_history'].append(tree_size)
        self.debug_metrics['simulation_step_markers'].append(sim_step)
        
        # 2. Root children visit counts
        root_children_indices = tree.children_index[sample_idx, BatchedMctsTree.ROOT_INDEX]  # [branch_k]
        root_children_visits = []
        for child_idx in root_children_indices:
            if child_idx != BatchedMctsTree.UNVISITED:
                visits = tree.node_visits[sample_idx, child_idx].item()
                root_children_visits.append(visits)
        self.debug_metrics['root_children_visits'].append(root_children_visits.copy())
        
        # 3. Reward distribution - collect rewards ONLY from newly evaluated nodes in this step
        # Also track terminal status, timesteps, and whether nodes are truly new
        num_terminal_nodes = 0
        num_reexpanded_nodes = 0
        truly_new_node_indices = []
        node_timesteps = []  # Track timesteps of expanded nodes
        
        if newly_evaluated_nodes is not None:
            # newly_evaluated_nodes shape: [batch_size, K] - only look at sample 0
            new_node_indices = newly_evaluated_nodes[sample_idx].cpu().numpy()  # [K]
            # Get unique node indices (terminal nodes might be repeated)
            unique_new_nodes = np.unique(new_node_indices)
            
            if len(unique_new_nodes) > 0:
                # Convert to torch tensor for indexing
                unique_new_nodes_tensor = torch.from_numpy(unique_new_nodes).to(device)
                
                # Check terminal status, visit counts, and timesteps for each node
                for node_idx in unique_new_nodes_tensor:
                    is_terminal = tree.is_terminal[sample_idx, node_idx].item()
                    visit_count = tree.node_visits[sample_idx, node_idx].item()
                    timestep_int = tree.node_timesteps_int[sample_idx, node_idx].item()
                    
                    node_timesteps.append(timestep_int)
                    
                    if is_terminal:
                        num_terminal_nodes += 1
                    
                    # A node is "truly new" if it was just evaluated for the first time
                    # For terminal nodes that are re-selected, visit_count > 1
                    if visit_count == 1:
                        truly_new_node_indices.append(node_idx.item())
                    else:
                        num_reexpanded_nodes += 1
                
                # Get rewards for these nodes
                current_step_rewards = tree.node_rewards[sample_idx, unique_new_nodes_tensor]
                # Filter out nodes with visit count = 0 (shouldn't happen, but safety check)
                valid_mask = tree.node_visits[sample_idx, unique_new_nodes_tensor] > 0
                if valid_mask.any():
                    valid_rewards = current_step_rewards[valid_mask]
                    if valid_rewards.numel() > 0:
                        self.debug_metrics['reward_history'].extend(valid_rewards.cpu().tolist())
                        valid_rewards_for_stats = valid_rewards
                    else:
                        valid_rewards_for_stats = torch.tensor([], device=device)
                else:
                    valid_rewards_for_stats = torch.tensor([], device=device)
            else:
                valid_rewards_for_stats = torch.tensor([], device=device)
        else:
            # Fallback: collect all non-zero rewards (old behavior)
            valid_mask = (tree.node_visits[sample_idx] > 0)
            valid_rewards = tree.node_rewards[sample_idx][valid_mask]
            if valid_rewards.numel() > 0:
                self.debug_metrics['reward_history'].extend(valid_rewards.cpu().tolist())
            valid_rewards_for_stats = valid_rewards if valid_rewards.numel() > 0 else torch.tensor([], device=device)
        
        # 4. Q-values of top nodes (sorted by visit count)
        visits = tree.node_visits[sample_idx]
        values = tree.node_values[sample_idx]
        Q_values = torch.where(visits > 0, values / visits.float(), torch.zeros_like(values))
        
        # Get top-10 most visited nodes
        valid_mask_all = (tree.node_visits[sample_idx] > 0)
        top_k = min(10, valid_mask_all.sum().item())
        if top_k > 0:
            top_visits, top_indices = torch.topk(visits, k=top_k)
            top_Q = Q_values[top_indices]
            self.debug_metrics['q_value_history'].append({
                'step': sim_step,
                'top_visits': top_visits.cpu().tolist(),
                'top_Q': top_Q.cpu().tolist()
            })
        
        # Log summary
        if self.global_rank == 0:
            logging.info(f"\n{'='*80}")
            logging.info(f"[DEBUG METRICS] Simulation Step {sim_step}")
            logging.info(f"{'='*80}")
            logging.info(f"  Tree Size: {tree_size} nodes")
            logging.info(f"  Root Children Visits: {root_children_visits}")
            
            if len(root_children_visits) > 0:
                max_visits = max(root_children_visits)
                min_visits = min(root_children_visits)
                balance_ratio = min_visits / max_visits if max_visits > 0 else 0.0
                logging.info(f"  Visit Balance Ratio (min/max): {balance_ratio:.3f}")
                if balance_ratio < 0.1:
                    logging.info(f"  ⚠️  WARNING: Highly imbalanced exploration! Consider increasing c_puct.")
            
            # Log expansion information
            if newly_evaluated_nodes is not None:
                total_evaluated = len(unique_new_nodes) if 'unique_new_nodes' in locals() else 0
                
                # Calculate average timestep of expanded nodes
                avg_timestep = np.mean(node_timesteps) if len(node_timesteps) > 0 else 0
                min_timestep = min(node_timesteps) if len(node_timesteps) > 0 else 0
                max_timestep = max(node_timesteps) if len(node_timesteps) > 0 else 0
                
                logging.info(f"  Expansion Info:")
                logging.info(f"    - Total unique nodes processed: {total_evaluated}")
                logging.info(f"    - Terminal nodes: {num_terminal_nodes}")
                logging.info(f"    - Re-expanded nodes (visit_count > 1): {num_reexpanded_nodes}")
                logging.info(f"    - Truly new nodes (visit_count = 1): {len(truly_new_node_indices)}")
                logging.info(f"    - Timestep stats (t_int): avg={avg_timestep:.1f}, min={min_timestep}, max={max_timestep}")
                logging.info(f"    - Normalized depth: {avg_timestep/self.T:.2%} of total diffusion steps")
                
                # Log temperature sampling configuration
                temp_values = self.mcts_config['temperature_values']
                logging.info(f"    - Temperature values: {[f'{t:.2f}' for t in temp_values]}")
                
                # Store in debug metrics for visualization
                if not hasattr(self.debug_metrics, 'expansion_timesteps_history'):
                    self.debug_metrics['expansion_timesteps_history'] = []
                self.debug_metrics['expansion_timesteps_history'].append({
                    'step': sim_step,
                    'avg_timestep': avg_timestep,
                    'min_timestep': min_timestep,
                    'max_timestep': max_timestep,
                    'timesteps': node_timesteps.copy()
                })
                
                if num_terminal_nodes > 0:
                    logging.info(f"  ⚠️  MCTS selected {num_terminal_nodes} terminal node(s) - these are already complete molecules!")
                
                if num_reexpanded_nodes > 0:
                    logging.info(f"  ⚠️  MCTS re-expanded {num_reexpanded_nodes} previously visited node(s)")
            
            # Log reward stats for newly evaluated nodes only
            if valid_rewards_for_stats.numel() > 0:
                num_new_nodes = valid_rewards_for_stats.numel()
                logging.info(f"  Reward Stats (Current Step - {num_new_nodes} evaluated nodes):")
                logging.info(f"    - Mean: {valid_rewards_for_stats.mean().item():.4f}")
                logging.info(f"    - Std:  {valid_rewards_for_stats.std().item():.4f}")
                logging.info(f"    - Min:  {valid_rewards_for_stats.min().item():.4f}")
                logging.info(f"    - Max:  {valid_rewards_for_stats.max().item():.4f}")
                
                # Check for sparse rewards
                near_zero = (valid_rewards_for_stats.abs() < 0.01).sum().item()
                total_rewards = valid_rewards_for_stats.numel()
                sparsity_ratio = near_zero / total_rewards
                logging.info(f"    - Sparsity (near-zero): {sparsity_ratio:.2%}")
                if sparsity_ratio > 0.9:
                    logging.info(f"  ⚠️  WARNING: Very sparse rewards! MCTS may struggle to learn.")
            else:
                logging.info(f"  Reward Stats: No nodes evaluated in this step")
            
            if top_k > 0:
                logging.info(f"  Top-{top_k} Nodes by Visits:")
                for i in range(min(5, top_k)):  # Show top 5
                    logging.info(f"    Node {top_indices[i].item()}: "
                               f"visits={top_visits[i].item()}, Q={top_Q[i].item():.4f}")
            logging.info(f"{'='*80}\n")
    
    def _log_final_debug_summary(self, tree: BatchedMctsTree, results: List[List[Tuple]], batch_size: int):
        """
        Log final summary including diversity analysis and search path visualization.
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"[FINAL DEBUG SUMMARY]")
        logging.info(f"{'='*80}")
        
        # Analyze first sample in detail
        sample_idx = 0
        if sample_idx < len(results):
            sample_results = results[sample_idx]
            
            logging.info(f"\nSample {sample_idx} Analysis:")
            logging.info(f"  Total molecules generated: {len(sample_results)}")
            
            if len(sample_results) > 0:
                # Diversity analysis
                unique_smiles = set(smi for smi, _, _ in sample_results)
                logging.info(f"  Unique molecules: {len(unique_smiles)}")
                logging.info(f"  Diversity ratio: {len(unique_smiles) / len(sample_results):.2%}")
                
                if len(unique_smiles) / len(sample_results) < 0.5:
                    logging.info(f"  ⚠️  WARNING: Low diversity! MCTS may be stuck in local optimum.")
                
                # Score distribution
                scores = [score for _, score, _ in sample_results]
                logging.info(f"  Score distribution:")
                logging.info(f"    - Mean: {np.mean(scores):.4f}")
                logging.info(f"    - Std:  {np.std(scores):.4f}")
                logging.info(f"    - Min:  {np.min(scores):.4f}")
                logging.info(f"    - Max:  {np.max(scores):.4f}")
                
                # Show top-5 molecules
                logging.info(f"\n  Top-5 molecules:")
                for i, (smi, score, mol) in enumerate(sample_results[:5]):
                    logging.info(f"    {i+1}. SMILES: {smi}")
                    logging.info(f"       Score: {score:.4f}")
        
        # Tree growth visualization
        if len(self.debug_metrics['tree_size_history']) > 0:
            logging.info(f"\n  Tree Growth Over Time:")
            steps = self.debug_metrics['simulation_step_markers']
            sizes = self.debug_metrics['tree_size_history']
            
            # Sample every 10th point for compact display
            sample_rate = max(1, len(steps) // 10)
            for i in range(0, len(steps), sample_rate):
                logging.info(f"    Step {steps[i]:4d}: {sizes[i]:5d} nodes")
            
            # Check if tree is growing steadily
            if len(sizes) > 1:
                growth_rate = (sizes[-1] - sizes[0]) / len(sizes)
                logging.info(f"  Average growth rate: {growth_rate:.2f} nodes/checkpoint")
                if growth_rate < 1.0:
                    logging.info(f"  ⚠️  WARNING: Tree growth stalled! Check terminal node handling.")
        
        # Reward distribution histogram
        if len(self.debug_metrics['reward_history']) > 0:
            logging.info(f"\n  Reward Distribution Histogram:")
            rewards = np.array(self.debug_metrics['reward_history'])
            
            # Create histogram
            hist, bin_edges = np.histogram(rewards, bins=10)
            for i in range(len(hist)):
                bar = '█' * int(hist[i] / max(hist) * 50) if max(hist) > 0 else ''
                logging.info(f"    [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {bar} ({hist[i]})")
        
        # Q-value trends
        if len(self.debug_metrics['q_value_history']) > 0:
            logging.info(f"\n  Q-value Trends (Top Node):")
            for entry in self.debug_metrics['q_value_history'][::max(1, len(self.debug_metrics['q_value_history'])//5)]:
                step = entry['step']
                top_Q = entry['top_Q'][0] if len(entry['top_Q']) > 0 else 0.0
                logging.info(f"    Step {step:4d}: Q = {top_Q:.4f}")
        
        logging.info(f"{'='*80}\n")
    
    def _save_debug_metrics(self):
        """
        Save debug metrics to a pickle file for later analysis.
        
        Creates a timestamped file in the current output directory.
        """
        import pickle
        from pathlib import Path
        
        # Create debug_metrics directory in the current output directory
        output_dir = Path.cwd() / 'debug_metrics'
        output_dir.mkdir(exist_ok=True)
        
        # Generate timestamped filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"mcts_debug_metrics_{timestamp}.pkl"
        filepath = output_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.debug_metrics, f)
            logging.info(f"\n[DEBUG] Metrics saved to: {filepath}")
            logging.info(f"[DEBUG] To visualize: python examples/analyze_debug_metrics.py {filepath}\n")
        except Exception as e:
            logging.warning(f"[DEBUG] Failed to save metrics: {e}")
    
    def _save_tree_structure(self, tree: BatchedMctsTree, sample_idx: int = 0):
        """
        Save tree structure for visualization.
        
        Exports tree topology, node properties, and relationships to JSON
        for external visualization tools.
        
        Args:
            tree: BatchedMctsTree after search completion
            sample_idx: Which sample's tree to save (default: 0)
        """
        import json
        from pathlib import Path
        
        output_dir = Path.cwd() / 'debug_metrics'
        output_dir.mkdir(exist_ok=True)
        
        # Collect tree data for the specified sample
        num_nodes = tree.num_nodes[sample_idx].item()
        
        nodes = []
        edges = []
        
        for node_id in range(num_nodes):
            if tree.node_visits[sample_idx, node_id].item() == 0:
                continue  # Skip unvisited nodes
            
            # Node properties
            node_data = {
                'id': int(node_id),
                'visits': int(tree.node_visits[sample_idx, node_id].item()),
                'value': float(tree.node_values[sample_idx, node_id].item()),
                'reward': float(tree.node_rewards[sample_idx, node_id].item()),
                'q_value': float(tree.node_values[sample_idx, node_id].item() / max(1, tree.node_visits[sample_idx, node_id].item())),
                'timestep': int(tree.node_timesteps_int[sample_idx, node_id].item()),
                'is_terminal': bool(tree.is_terminal[sample_idx, node_id].item()),
            }
            nodes.append(node_data)
            
            # Parent-child edges
            parent_id = tree.parents[sample_idx, node_id].item()
            if parent_id != BatchedMctsTree.NO_PARENT and parent_id >= 0:
                edges.append({
                    'source': int(parent_id),
                    'target': int(node_id),
                })
        
        tree_data = {
            'sample_idx': sample_idx,
            'num_nodes': num_nodes,
            'num_visited': len(nodes),
            'nodes': nodes,
            'edges': edges,
            'config': {
                'branch_k': tree.branch_k,
                'c_puct': self.mcts_config['c_puct'],
                'expand_steps': self.mcts_config['expand_steps'],
                'prediffuse_steps': self.mcts_config['prediffuse_steps'],
            }
        }
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"mcts_tree_structure_{timestamp}_sample{sample_idx}.json"
        filepath = output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(tree_data, f, indent=2)
            logging.info(f"[DEBUG] Tree structure saved to: {filepath}")
            logging.info(f"[DEBUG] To visualize: python examples/visualize_tree.py {filepath}")
        except Exception as e:
            logging.warning(f"[DEBUG] Failed to save tree structure: {e}")
    
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
    def _batched_prediffuse(self, tree: BatchedMctsTree, prediffuse_steps: int) -> BatchedMctsTree:
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
        t_thresh = max(0, self.T - prediffuse_steps)
        for s_int in reversed(range(t_thresh, self.T)):
            # s_array: [batch_size, 1]
            s_array = torch.full((batch_size, 1), s_int, dtype=torch.float32, device=device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            
            # Batched denoising: sample_p_zs_given_zt handles batch dimension
            # Use temperature=1.0 (standard sampling, no diversity needed in prediffuse)
            sampled_s, _ = self.sample_p_zs_given_zt(s_norm, t_norm, X_cur, E_cur, y_cur, mask_cur, temperature=1.0)
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
            uct_scores = torch.where(is_invalid, torch.tensor(-1 * float('inf'), device=device), uct_scores) # the inf does not matter
            
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
        - Perform multi-step denoising (expand_steps) with dynamic masking:
          * For each step, denoise only active samples (timestep > 0)
          * Samples that reach terminal state (t=0) stop denoising
          * Continue until all samples reach terminal or expand_steps exhausted
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
        
        # Multi-step denoising with dynamic masking
        # Get number of expansion steps from config
        expand_steps = self.mcts_config.get('expand_steps', 1)
        
        # Prepare temperature values for diversity in MCTS expansion
        # Get temperature values (always a list of length K, validated in _init_mcts_config)
        temp_values = self.mcts_config['temperature_values']
        
        # Create [B*K] temperature tensor to match the interleaved batch structure
        # The expanded states are created via repeat_interleave(K, dim=0), which gives:
        #   [sample0_child0, sample0_child1, ..., sample0_childK-1, 
        #    sample1_child0, sample1_child1, ..., sample1_childK-1, ...]
        # We create matching temperature pattern:
        #   [temp[0], temp[1], ..., temp[K-1], temp[0], temp[1], ..., temp[K-1], ...]
        # This ensures child i always uses temp_values[i] for diversity
        temp_list = []
        for b in range(batch_size):
            temp_list.extend(temp_values)  # Add all K temperatures for this sample
        temperatures = torch.tensor(temp_list, device=device, dtype=torch.float32)  # [B*K]
        
        # Track current timesteps for each sample
        current_t_int = expanded_t_int # [B*K]
        current_X = expanded_X # [B*K, n_atoms, X_dim]
        current_E = expanded_E  # [B*K, n_atoms, n_atoms, E_dim]
        
        # Perform expand_steps denoising steps with dynamic masking
        for step in range(expand_steps):
            # Create active mask: which samples still need denoising (current_t_int > 0)
            active_mask = current_t_int > 0  # [B*K]
            
            if not active_mask.any():
                # All samples have reached terminal state
                break
            
            # Compute next timestep: t' = t - 1
            next_t_int = (current_t_int - 1).clamp(min=0)
            
            # Prepare s_norm and t_norm for denoising
            # For child at timestep t', denoising uses s'=(t'-1)/T, t'=(t')/T
            # Shape: [B*K, 1]
            next_leaf_t_norm = (current_t_int.float() / self.T).unsqueeze(1)  # [B*K, 1]
            next_t_norm = (next_t_int.float() / self.T).unsqueeze(1)  # [B*K, 1]
            
            # Batched denoising: sample K children per sample in one forward pass
            # Input: [B*K, ...] where batch contains interleaved children
            # temperature: [B*K] tensor where each child gets its designated temperature
            # This creates diversity: child 0 uses temp[0], child 1 uses temp[1], etc.
            sampled_states, _ = self.sample_p_zs_given_zt(
                next_t_norm, next_leaf_t_norm, current_X, current_E, expanded_y, expanded_mask,
                temperature=temperatures
            )
            
            # Update ONLY active samples using the mask
            # For inactive samples (already at t=0), keep their current state
            active_mask_E = active_mask.view(-1, 1, 1, 1)  # [B*K, 1, 1, 1]
            current_E = torch.where(active_mask_E, sampled_states.E, current_E)
            
            # Note: X stays unchanged (edge-only denoising)
            # current_X remains the same
            
            # Update timesteps for active samples
            current_t_int = torch.where(active_mask, next_t_int, current_t_int)
        
        # After multi-step denoising, use the final states
        next_t_int = current_t_int  # [B*K]
        sampled_E = current_E  # [B*K, n_atoms, n_atoms, E_dim]
        
        # Compute final normalized timesteps
        next_t_norm_final = (next_t_int.float() / self.T)  # [B*K]
        next_s_norm_final = ((next_t_int.float() - 1).clamp(min=0) / self.T)  # [B*K]
        
        # Reshape back to [batch_size, K, ...]
        sampled_E_reshaped = sampled_E.view(batch_size, K, *sampled_E.shape[1:])  # [B, K, n_atoms, n_atoms, E_dim]
        original_X_reshaped = expanded_X.view(batch_size, K, *expanded_X.shape[1:])  # [B, K, n_atoms, X_dim]
        next_t_int_reshaped = next_t_int.view(batch_size, K)  # [B, K]
        # Note: next_t_norm_final and next_s_norm_final are [B*K], when reshaped they become [B, K]
        next_t_norm_reshaped = next_t_norm_final.view(batch_size, K)  # [B, K]
        next_s_norm_reshaped = next_s_norm_final.view(batch_size, K)  # [B, K]
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
        spectra: List[np.ndarray],
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
            # In _batched_evaluate, deduplicate before scoring
            # This is the key optimization: one verifier call instead of N calls
            self._ensure_verifier()
            current_time = time.time()
            unique_smiles = {}
            from collections import defaultdict
            mol_indices_map = defaultdict(list)

            for i, (mol, smi) in enumerate(zip(mol_list, smi_list)):
                if smi not in unique_smiles:
                    unique_smiles[smi] = (mol, smi, precursor_mzs[i], adducts[i], instruments[i], collision_engs[i], target_specs[i])
                mol_indices_map[smi].append(i)

            # Score only unique SMILES
            unique_scores = self.verifier.score_batch(*zip(*list(unique_smiles.values())), bin_size=self.cfg.mcts.similarity.bin_size)

            # Remap scores to original indices
            batch_scores = torch.zeros(N, device=device)
            for smi, score in zip(unique_smiles.keys(), unique_scores):
                for original_idx in mol_indices_map[smi]:
                    batch_scores[original_idx] = score

            # logging.info(f"Time taken for scoring: {time.time() - current_time} seconds")

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
    
    def _merge_distributed_predictions(self, stage: str = 'test'):
        """
        Merge prediction files from all GPU ranks into unified outputs.
        
        This function should only be called on rank 0 after all ranks have finished
        saving their predictions. It collects all rank-specific pickle files and
        merges them in the correct order.
        
        Args:
            stage: 'test' or 'val' to determine which files to merge
            
        Output files:
            - preds/{name}_{stage}_pred_merged.pkl: All predictions merged
            - preds/{name}_{stage}_true_merged.pkl: All ground truths merged
        """
        import glob
        from pathlib import Path
        
        pred_dir = Path('preds')
        if not pred_dir.exists():
            logging.warning(f"Predictions directory does not exist: {pred_dir}")
            return
        
        # Determine number of ranks by finding all rank files
        # Pattern: {name}_rank_{rank}_pred_{batch_idx}.pkl
        rank_files = sorted(glob.glob(str(pred_dir / f"{self.name}_rank_*_pred_*.pkl")))
        
        if len(rank_files) == 0:
            logging.warning(f"No rank-specific prediction files found for merging")
            return
        
        # Extract unique ranks and batch indices
        ranks = set()
        batch_indices = set()
        for file_path in rank_files:
            file_name = Path(file_path).stem  # Remove .pkl extension
            # Parse: {name}_rank_{rank}_pred_{batch_idx}
            parts = file_name.split('_')
            # Find 'rank' keyword and extract rank number
            try:
                rank_idx = parts.index('rank')
                rank = int(parts[rank_idx + 1])
                # Find 'pred' keyword and extract batch index
                pred_idx = parts.index('pred')
                batch_idx = int(parts[pred_idx + 1])
                ranks.add(rank)
                batch_indices.add(batch_idx)
            except (ValueError, IndexError) as e:
                logging.warning(f"Could not parse file name: {file_path}, error: {e}")
                continue
        
        ranks = sorted(ranks)
        batch_indices = sorted(batch_indices)
        
        if not ranks or not batch_indices:
            logging.warning("Could not determine ranks or batch indices from file names")
            return
        
        logging.info(f"=" * 80)
        logging.info(f"[POST-PROCESSING] Merging distributed predictions for {stage}")
        logging.info(f"  Found {len(ranks)} ranks: {ranks}")
        logging.info(f"  Found {len(batch_indices)} batches: {batch_indices}")
        logging.info(f"=" * 80)
        
        # Merge predictions and ground truths
        all_predictions = []
        all_ground_truths = []
        
        # Process batches in order, then ranks in order
        for batch_idx in batch_indices:
            for rank in ranks:
                pred_file = pred_dir / f"{self.name}_rank_{rank}_pred_{batch_idx}.pkl"
                true_file = pred_dir / f"{self.name}_rank_{rank}_true_{batch_idx}.pkl"
                
                if pred_file.exists() and true_file.exists():
                    try:
                        with open(pred_file, 'rb') as f:
                            batch_preds = pickle.load(f)
                        with open(true_file, 'rb') as f:
                            batch_trues = pickle.load(f)
                        
                        # Each file contains a list of predictions/truths for samples in that batch
                        all_predictions.extend(batch_preds)
                        all_ground_truths.extend(batch_trues)
                        
                        logging.info(f"  Loaded rank {rank}, batch {batch_idx}: "
                                   f"{len(batch_preds)} samples")
                    except Exception as e:
                        logging.warning(f"Failed to load {pred_file} or {true_file}: {e}")
                else:
                    if not pred_file.exists():
                        logging.warning(f"Missing prediction file: {pred_file}")
                    if not true_file.exists():
                        logging.warning(f"Missing ground truth file: {true_file}")
        
        # Save merged results
        if len(all_predictions) > 0:
            merged_pred_file = pred_dir / f"{self.name}_{stage}_pred_merged.pkl"
            merged_true_file = pred_dir / f"{self.name}_{stage}_true_merged.pkl"
            
            try:
                with open(merged_pred_file, 'wb') as f:
                    pickle.dump(all_predictions, f)
                with open(merged_true_file, 'wb') as f:
                    pickle.dump(all_ground_truths, f)
                
                logging.info(f"=" * 80)
                logging.info(f"[POST-PROCESSING] Merge complete!")
                logging.info(f"  Total samples merged: {len(all_predictions)}")
                logging.info(f"  Saved to:")
                logging.info(f"    - {merged_pred_file}")
                logging.info(f"    - {merged_true_file}")
                logging.info(f"=" * 80)
                
                # Optionally, remove rank-specific files to save space
                # Uncomment the following lines if you want to auto-cleanup
                # for batch_idx in batch_indices:
                #     for rank in ranks:
                #         pred_file = pred_dir / f"{self.name}_rank_{rank}_pred_{batch_idx}.pkl"
                #         true_file = pred_dir / f"{self.name}_rank_{rank}_true_{batch_idx}.pkl"
                #         if pred_file.exists():
                #             pred_file.unlink()
                #         if true_file.exists():
                #             true_file.unlink()
                # logging.info("  Cleaned up rank-specific files")
                
            except Exception as e:
                logging.error(f"Failed to save merged predictions: {e}")
        else:
            logging.warning("No predictions to merge!")
    
    def _merge_distributed_caches(self):
        """
        Merge verifier caches from all GPU ranks into unified cache files.
        
        This function should only be called on rank 0 after all ranks have finished
        their inference. It collects all rank-specific cache files and merges them
        to avoid redundant computation in future runs.
        
        Output files:
            - cache/mcts/spectra_cache_merged.pkl (for IcebergVerifier)
            - cache/mcts/scores_cache_merged.pkl (for IcebergVerifier)
            - cache/mcts/graff_spectra_cache_merged.pkl (for GraffMSVerifier)
            - cache/mcts/graff_scores_cache_merged.pkl (for GraffMSVerifier)
        """
        import glob
        from pathlib import Path
        
        # Determine cache directory from config
        cache_dir = Path(getattr(self.cfg.mcts, 'cache_dir', './cache/mcts/'))
        if not cache_dir.exists():
            logging.warning(f"Cache directory does not exist: {cache_dir}")
            return
        
        # Determine verifier type
        verifier_type = getattr(self.cfg.mcts, 'verifier_type', 'iceberg')
        
        if verifier_type == 'iceberg':
            spectra_pattern = 'spectra_cache_rank*.pkl'
            scores_pattern = 'scores_cache_rank*.pkl'
            merged_spectra_name = 'spectra_cache_merged.pkl'
            merged_scores_name = 'scores_cache_merged.pkl'
        elif verifier_type == 'graffms':
            spectra_pattern = 'graff_spectra_cache_rank*.pkl'
            scores_pattern = 'graff_scores_cache_rank*.pkl'
            merged_spectra_name = 'graff_spectra_cache_merged.pkl'
            merged_scores_name = 'graff_scores_cache_merged.pkl'
        else:
            logging.warning(f"Unknown verifier type: {verifier_type}, skipping cache merge")
            return
        
        logging.info(f"=" * 80)
        logging.info(f"[CACHE MERGE] Merging {verifier_type} caches from all ranks")
        logging.info(f"=" * 80)
        
        # Find all rank-specific cache files
        spectra_files = sorted(glob.glob(str(cache_dir / spectra_pattern)))
        scores_files = sorted(glob.glob(str(cache_dir / scores_pattern)))
        
        if not spectra_files and not scores_files:
            logging.info("No rank-specific cache files found to merge")
            return
        
        logging.info(f"  Found {len(spectra_files)} spectra cache files")
        logging.info(f"  Found {len(scores_files)} scores cache files")
        
        # Merge spectra caches
        merged_spectra = {}
        for file_path in spectra_files:
            try:
                with open(file_path, 'rb') as f:
                    rank_cache = pickle.load(f)
                # Update with rank cache (later ranks overwrite earlier ones if duplicate keys)
                merged_spectra.update(rank_cache)
                logging.info(f"  Loaded {len(rank_cache)} entries from {Path(file_path).name}")
            except Exception as e:
                logging.warning(f"Failed to load {file_path}: {e}")
        
        # Merge scores caches
        merged_scores = {}
        for file_path in scores_files:
            try:
                with open(file_path, 'rb') as f:
                    rank_cache = pickle.load(f)
                merged_scores.update(rank_cache)
                logging.info(f"  Loaded {len(rank_cache)} entries from {Path(file_path).name}")
            except Exception as e:
                logging.warning(f"Failed to load {file_path}: {e}")
        
        # Save merged caches
        if merged_spectra:
            merged_spectra_path = cache_dir / merged_spectra_name
            try:
                with open(merged_spectra_path, 'wb') as f:
                    pickle.dump(merged_spectra, f, protocol=4)
                logging.info(f"  Saved merged spectra cache: {len(merged_spectra)} entries -> {merged_spectra_path}")
            except Exception as e:
                logging.error(f"Failed to save merged spectra cache: {e}")
        
        if merged_scores:
            merged_scores_path = cache_dir / merged_scores_name
            try:
                with open(merged_scores_path, 'wb') as f:
                    pickle.dump(merged_scores, f, protocol=4)
                logging.info(f"  Saved merged scores cache: {len(merged_scores)} entries -> {merged_scores_path}")
            except Exception as e:
                logging.error(f"Failed to save merged scores cache: {e}")
        
        logging.info(f"=" * 80)
        logging.info(f"[CACHE MERGE] Complete!")
        logging.info(f"  Total unique spectra cached: {len(merged_spectra)}")
        logging.info(f"  Total unique scores cached: {len(merged_scores)}")
        logging.info(f"=" * 80)
