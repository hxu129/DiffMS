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
from typing import Optional, List, Dict, Tuple
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

    def _mcts_select(self, root: MctsNode, c_puct: float) -> Tuple[list, MctsNode]:
        # Only select among current leaf nodes using UCT; return (path_to_leaf, selected_leaf)
        # First, traverse to collect all leaves with their path
        stack = [(root, [root])]
        leaves = []
        while stack:
            node, path = stack.pop()
            if not node.children or node.terminal:
                leaves.append((path, node))
            else:
                for ch in node.children:
                    stack.append((ch, path + [ch]))
        # Compute U for each leaf w.r.t its parent
        best = None
        best_score = -1e9
        for path, leaf in leaves:
            # UCT: V + c * sqrt(ln N(parent) / N(leaf))
            if leaf.N == 0:
                leaf.U = float('inf')
                return (path, leaf)
            else:
                parent_N = max(1, leaf.parent.N)
                leaf_U = c_puct * math.sqrt(math.log(parent_N) / leaf.N)
                leaf.U = leaf_U
                score = leaf.V + leaf.U
            if score > best_score:
                best_score = score
                best = (path, leaf)
        return best

    def _mcts_expand(self, leaf: MctsNode, K: int) -> List[MctsNode]:
        if leaf.terminal:
            return []
        state = leaf.state
        t_int = int(state['t_int'])
        if t_int <= 0:
            leaf.terminal = True
            return []
        # Sample K next states by calling sample_p_zs_given_zt K times
        leaf.children = []
        seen = set()
        for idx in range(K):
            s_norm = state['s_norm']
            t_norm = state['t_norm']
            X_t = state['X_t']
            E_t = state['E_t']
            y = state['y_t']
            node_mask = state['node_mask']
            sampled_one_hot, _ = self.sample_p_zs_given_zt(s_norm, t_norm, X_t, E_t, y, node_mask)
            E_next = sampled_one_hot.E
            # uniqueness via hash of discrete E
            # e_idx = torch.argmax(sampled_one_hot.E, dim=-1).detach().cpu().numpy()
            # h = e_idx.tobytes()
            # if h in seen:
            #     continue
            # seen.add(h)
            next_t = t_int - 1
            # For the child state, its next sampling call will use s'=(next_t-1)/T, t'=(next_t)/T
            next_t_norm = torch.tensor([[next_t]], dtype=torch.float32, device=self.device) / self.T
            next_s_norm = torch.tensor([[max(0, next_t - 1)]], dtype=torch.float32, device=self.device) / self.T
            child = MctsNode(
                state={
                    't_int': next_t,
                    's_norm': next_s_norm,
                    't_norm': next_t_norm,
                    # edge-only denoising: keep X_t unchanged
                    'X_t': X_t,
                    'E_t': E_next,
                    'y_t': y,
                    'node_mask': node_mask,
                },
                children=[], parent=leaf,
                terminal=(next_t == 0),
                N=0, V=0.0, r=0.0, best_smiles=None, best_score=-1e9,
            )
            leaf.children.append(child)
        # No prior needed under classic UCT; children share parent equally via ln N(parent)
        return leaf.children

    def _mcts_evaluate(self, node: MctsNode, env_meta: dict, spectra: np.ndarray) -> float:
        state = node.state
        X_t = state['X_t']
        E_t = state['E_t'] 
        y = state['y_t']
        node_mask = state['node_mask']
        t_int = int(state['t_int'])
        score = 0.0
        if t_int == 0:
            node.terminal = True
            sampled_s = utils.PlaceHolder(X=X_t, E=E_t, y=y)
            sampled_s = sampled_s.mask(node_mask, collapse=True)
            X_hat = sampled_s.X
            E_hat = sampled_s.E
        else:
            # ------- start of denoising step -------
            noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y, 't': state['t_norm'], 'node_mask': node_mask}
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            prob_E = F.softmax(pred.E, dim=-1)
            prob_X = F.softmax(pred.X, dim=-1)
            assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
            assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

            sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
            X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
            E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

            assert (E_s == torch.transpose(E_s, 1, 2)).all()
            assert (X_s.shape == X_t.shape) and (E_s.shape == E_t.shape)

            sampled_s = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y.shape[0], 0)).mask(node_mask).type_as(y)
            # ------- end of denoising step -------

            sampled_s.X = X_t
            sampled_s = sampled_s.mask(node_mask, collapse=True)
            X_hat = sampled_s.X
            E_hat = sampled_s.E

        X_hat = X_hat.squeeze(0) # batch size = 1, remove the batch dim
        E_hat = E_hat.squeeze(0) # batch size = 1, remove the batch dim
        
        valid, smi, mol = self._terminal_check_and_smiles(X_hat, E_hat)
        if not valid:
            return -1.0
        self._ensure_verifier()
        if smi in self._smiles_score_cache:
            score = float(self._smiles_score_cache[smi])
        else:
            # FIXME: why use smiles to generate score?
            score_list = self.verifier.score([mol], [smi], 
                                                env_meta['precursor_mz'], env_meta['adduct'], 
                                                env_meta['instrument'], env_meta['collision_eng'], spectra)
            # FIXME: the shape is disgusting
            score = float(score_list[0])
            self._smiles_score_cache[smi] = score
        if node.parent is not None and (node.best_smiles is None or score > node.best_score):
            node.best_smiles = smi
            node.best_score = score
        return score

    def _mcts_backup(self, path: List[MctsNode], value: float) -> None:
        # Update NVrU along the path. Here value is already the evaluated result propagated upward.
        for node in path:
            node.r = float(value)  # last evaluation result viewed from this node
            node.N = int(node.N) + 1
            node.V = (node.V * (node.N - 1) + node.r) / max(1, node.N)
            # U is recomputed during selection; keep as informational

    def _mcts_sample_single(
        self,
        X_init: torch.Tensor, E_init: torch.Tensor, y: torch.Tensor,
        node_mask: torch.Tensor,
        env_meta: dict,
        spectra: np.ndarray,
        num_simulations: int,
        K: int,
        c_puct: float,
        t_thresh: int,
    ) -> List[Tuple[str, float, Chem.Mol]]:
        # First deffuse t_thresh times 
        for s_int in reversed(range(t_thresh, self.T)):
            s_array = s_int * torch.ones((X_init.shape[0], 1), dtype=torch.float32, device=self.device)
            # TODO: check the termination condition
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, __ = self.sample_p_zs_given_zt(s_norm, t_norm, X_init, E_init, y, node_mask)
            E_init = sampled_s.E

        # root state at t=T
        T = t_thresh
        # TODO: the shape of X and E seems to be wrong
        s_norm = torch.tensor([[T - 1]], dtype=torch.float32, device=self.device) / self.T
        t_norm = torch.tensor([[T]], dtype=torch.float32, device=self.device) / self.T
        root = MctsNode(
            state={'t_int': T, 's_norm': s_norm, 't_norm': t_norm, 'X_t': X_init, 'E_t': E_init, 'y_t': y, 'node_mask': node_mask},
            children=[], parent=None, terminal=(T == 0),
            N=0, V=0.0, r=0.0, best_smiles=None, best_score=-1e9,
        )
        # reset per-run results cache to avoid cross-sample leakage
        self._smiles_score_cache = {}
        for _ in range(int(num_simulations)):
            path, leaf = self._mcts_select(root, c_puct)
            # If not a leaf (or expandable), expand once to simulate forward then evaluate exactly one node
            if not leaf.terminal and not leaf.children:
                children = self._mcts_expand(leaf, K)
                # evaluate all children and backup individually
                for ch in children:
                    v = self._mcts_evaluate(ch, env_meta, spectra)
                    self._mcts_backup(path + [ch], v)
            else:
                v = self._mcts_evaluate(leaf, env_meta, spectra)
                self._mcts_backup(path + [leaf], v)

        # After search, prioritize true terminal leaves; then complete top non-terminal leaves to terminal
        def _collect_leaves(n: MctsNode):
            stack = [n]
            leaves = []
            while stack:
                cur = stack.pop()
                if not cur.children or cur.terminal:
                    leaves.append(cur)
                else:
                    stack.extend(cur.children)
            return leaves

        leaves = _collect_leaves(root)
        terminal_leaves = [ln for ln in leaves if int(ln.state['t_int']) == 0]
        nonterminal_leaves = [ln for ln in leaves if int(ln.state['t_int']) > 0]

        results: list[tuple[str, float, Chem.Mol]] = []
        seen_smi = set()

        # 1) add scored terminal leaves first (evaluate if needed)
        for ln in terminal_leaves:
            st = ln.state
            X_t = st['X_t']
            E_t = st['E_t']
            y = st['y_t']
            node_mask = st['node_mask']
            sampled_s = utils.PlaceHolder(X=X_t, E=E_t, y=y)
            sampled_s = sampled_s.mask(node_mask, collapse=True)
            X_hat = sampled_s.X.squeeze(0)
            E_hat = sampled_s.E.squeeze(0)
            valid, smi, mol = self._terminal_check_and_smiles(X_hat, E_hat)
            # if not valid or smi in seen_smi:
            #     continue
            self._ensure_verifier()
            if smi in self._smiles_score_cache:
                score = float(self._smiles_score_cache[smi])
            else:
                score = float(self._mcts_evaluate(ln, env_meta, spectra))
            results.append((smi, score, mol))
            seen_smi.add(smi)
            if len(results) >= self.test_num_samples:
                break

        # 2) fill remaining by completing best non-terminal leaves (by V) to terminal greedily
        if len(results) < self.test_num_samples and nonterminal_leaves:
            nonterminal_leaves.sort(key=lambda n_: n_.V, reverse=True)
            for ln in nonterminal_leaves:
                if len(results) >= self.test_num_samples:
                    break
                st = ln.state
                X_cur = st['X_t']  # keep nodes fixed (edge-only denoising)
                E_cur = st['E_t']
                y_cur = st['y_t']
                mask_cur = st['node_mask']
                t_cur = int(st['t_int'])
                # standard denoising from current t down to 0
                for s_int in reversed(range(0, t_cur)):
                    s_arr = torch.tensor([[s_int]], dtype=torch.float32, device=self.device) / self.T
                    t_arr = torch.tensor([[s_int + 1]], dtype=torch.float32, device=self.device) / self.T
                    sampled_s, _ = self.sample_p_zs_given_zt(s_arr, t_arr, X_cur, E_cur, y_cur, mask_cur)
                    # only update edges
                    E_cur = sampled_s.E
                # finalize X as current nodes (edge denoising only)
                sampled_s.X = X_cur
                sampled_s = sampled_s.mask(mask_cur, collapse=True)
                X_hat = sampled_s.X.squeeze(0)
                E_hat = sampled_s.E.squeeze(0)
                valid, smi, mol = self._terminal_check_and_smiles(X_hat, E_hat)
                # if not valid or smi in seen_smi:
                #     continue
                # score via verifier (or cache)
                self._ensure_verifier()
                if smi in self._smiles_score_cache:
                    score = float(self._smiles_score_cache[smi])
                else:
                    score_list = self.verifier.score([mol], [smi], 
                                                     env_meta['precursor_mz'], env_meta['adduct'], 
                                                     env_meta['instrument'], env_meta['collision_eng'], spectra)
                    score = float(score_list[0])
                    self._smiles_score_cache[smi] = score
                results.append((smi, score, mol))
                seen_smi.add(smi)

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.test_num_samples]

    @torch.no_grad()
    def mcts_sample_batch(self, data: Batch, env_metas: List[dict], spectra: List[np.ndarray]) -> List[List[Tuple[str, float, Chem.Mol]]]:
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X_all, E_all, y_all = dense_data.X, z_T.E, data.y

        assert (E_all == torch.transpose(E_all, 1, 2)).all()

        out = []
        bs = X_all.shape[0]
        for i in range(bs):
            X_i = X_all[i:i+1]
            E_i = E_all[i:i+1]
            y_i = y_all[i:i+1]
            mask_i = node_mask[i:i+1]
            metas = env_metas[i]
            spectra_i = spectra[i]
            res_i = self._mcts_sample_single(
                X_init=X_i, E_init=E_i, y=y_i, node_mask=mask_i, env_meta=metas,
                spectra=spectra_i,
                num_simulations=self.mcts_config['num_simulation_steps'],
                K=self.mcts_config['branch_k'], c_puct=self.mcts_config['c_puct'],
                t_thresh=self.mcts_config['t_thresh'],
            )
            out.append(res_i)
        return out

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
