#academic #generative-models #ai4science #mass-spectrum #monte-carlo-tree-search 

> This work is basically a new work of mine. 

# Background

Mass spectrum data is very important and useful in identifying molecules as well as characterizing molecular properties. However, solving the spectrum data requires lots of human experiences, and data centric methods are still inferior to the expers.

This paper is focusing on the **inverse** problems. Current approaches includes
- **Autoregressive** approaches, which are trained to convert tokenized spectral informations into SMILES strings as outputs. One representative methods is [[Spec2Mol]]. However, due to the **autoregressive nature**, these methods fail to capture the **permutation invariant** nature of spectral data, and they do not **incorporate other information about the molecules.**
- **Indirect** approaches, which uses intermediate representations of molecules before generating chemical structures. These methods are more chemically interpretable, but **not necessarily lead to better performance**.

A crucial problem making the inverse problem hard to solve is that, even for the same spectrum, there are possibly multiple molecule candidates which can satisfy the constrains of the spectrum well. Current methods directly uses **beam search**  or other naive methods like [[DiffMS]] and select top-k candidates with high probabilities, which fails to fully explore the whole output space, resulting in low novelty and low diversity.

To solve the **one-to-many** problem, we would like to more efficiently yet thoroughly explore the output space with some **heuristic** methods. Inspired by the success of Monte Carlo Tree Search in the protein inverse folding problems (to generate molecular structures given 3D structures) like [[ProtInvTree]], we would like to utilize the probability distribution learnt in the **discrete diffusion** model to do MCTS.

# Method

People usually generate the molecular structures by generating 
- **Fingerprints**, which is not exactly the structures, but **abstractions of molecular properties**, and the task is then modeled as a regression task -- to generate a high-dimensional vector, like what they do in [[MIST]];
- **[[SMILES]]**, which is the representation of molecular structures, and the problem becomes a sequence generation task, which is usually modeled in an autoregressive manner like [[Spec2Mol]]; and
- **Connectivity graph**, which uses connectivity graphs of atoms to directly represent the molecular structure, like [[DiffMS]].

To do Monte Carlo Tree Search, graph based representations seem to be superior for the permutation invariant property. In this work, we do MCTS based on [[DiffMS]], and the whole process can be divided into several steps:

## Overview: Batched MCTS Pipeline

The implementation uses **fully batched operations** to process multiple samples simultaneously, with the following stages:

1. **Initialization**: Create batched MCTS tree with root states at maximum noise (timestep T)
2. **Pre-diffusion**: Denoise root states from T down to `t_thresh = T - prediffuse_steps` 
3. **Main MCTS Loop** (repeat for `num_simulation_steps` iterations):
   - **Selection**: Pick one leaf per sample using UCT formula
   - **Expansion**: Create K children for each selected leaf via multi-step denoising
   - **Evaluation**: Score all children using batched verifier (ICEBERG)
   - **Backup**: Propagate scores up to root nodes
4. **Result Extraction**: Collect top-k molecules from tree, complete non-terminal nodes, and batch score all molecules

### Step 0: Feature Extraction

Uses the encoder module in DiffMS to extract spectral embeddings from input MS/MS data. The encoder converts raw spectral peaks `(m/z, intensity)` pairs into a fixed-dimensional embedding that conditions the diffusion model.

### Step 1: Initialization (`_initialize_batched_tree`)

**Input**: Batch of graph data with shape `[batch_size, n_atoms, ...]`

**Process**:
1. Convert input graphs to dense format: `X` (node features), `E` (edge features), `y` (global features)
2. Sample initial noise for edges from limit distribution (X stays fixed for edge-only denoising)
3. Allocate memory for batched tree structure: `[batch_size, max_nodes, ...]`
   - `max_nodes = num_simulation_steps × branch_k + 100` (buffer for safety)
4. Initialize root nodes (index 0) with:
   - States: `(X_init, E_noisy, y)`
   - Timestep: `t_int = T` (maximum noise level)
   - Visits: `N = 0`, Values: `V = 0`, Rewards: `r = 0`

**Output**: Batched tree with initialized roots at timestep T

### Step 2: Pre-diffusion (`_batched_prediffuse`)

**Purpose**: Reduce search space by pre-denoising roots to intermediate timestep

**Input**: Tree with roots at `t = T`

**Process**:
1. For each timestep `s` from `T-1` down to `t_thresh = T - prediffuse_steps`:
   - Denoise all root states in parallel: `p(E_s | E_{s+1}, X, y)`
   - Update edge states (X remains fixed)
2. Update root metadata:
   - `t_int = t_thresh`
   - `is_terminal = (t_thresh == 0)`

**Hyperparameter**: `prediffuse_steps` (default: 200)
- Higher values → start search closer to clean molecules (faster but less exploration)
- Lower values → more exploration but slower convergence

**Output**: Tree with roots at `t = t_thresh`

### Step 3: Selection (`_batched_select`)

**Purpose**: Traverse tree from root using UCT until finding nodes ready for expansion

**Input**: Tree, exploration constant `c_puct`

**Process** (vectorized across batch):
1. Start at root nodes for all samples
2. **While current nodes have children**:
   - For each child, compute UCT score:
     $$\text{UCT}(\text{child}) = Q(\text{child}) + c_{\text{puct}} \cdot \sqrt{\frac{\ln N(\text{parent})}{N(\text{child})}}$$
     where:
     - $Q(\text{child}) = V(\text{child}) / N(\text{child})$ (average value)
     - $N(\cdot)$ is visit count
     - Unvisited children (N=0) get infinite UCT for exploration
   - Select child with highest UCT score
   - Move to that child
3. **Stop when** reaching node that either:
   - Has no children yet (unexpanded leaf)
   - Is terminal (`t_int == 0`)

**Hyperparameter**: `c_puct` (default: 0.25)
- Higher values → more exploration (favor less-visited nodes)
- Lower values → more exploitation (favor high-value nodes)

**Output**: `[batch_size]` leaf node indices ready for expansion

### Step 4: Expansion (`_batched_expand`)

**Purpose**: Create K children for each selected leaf via multi-step denoising

**Input**: Tree, `[batch_size]` leaf indices, branch factor `K`

**Process**:
1. **Skip terminal or already-expanded nodes**:
   - Terminal nodes (`t_int == 0`): evaluate self K times (for batch consistency)
   - Already expanded nodes: skip expansion
   
2. **For expandable leaves**:
   - Allocate K consecutive node indices in tree
   - Gather leaf states: `(X_t, E_t, y_t, mask, t_int)`
   - Repeat each state K times → `[batch_size × K, ...]`
   
3. **Multi-step denoising with dynamic masking**:
   - For `expand_steps` iterations:
     - Create active mask: which samples have `current_t_int > 0`
     - For active samples: denoise one step `p(E_{t-1} | E_t, X, y)`
     - For inactive samples (reached t=0): keep current state
     - Update timesteps: `current_t_int = max(current_t_int - 1, 0)`
   - Continue until all samples reach terminal or `expand_steps` exhausted
   
4. **Store children in tree**:
   - States: `(X_parent, E_denoised, y_parent)`  (X unchanged, edge-only denoising)
   - Timesteps: `t_int = parent_t_int - expand_steps`
   - Parent-child relationships: update `children_index`, `parents`

**Hyperparameters**:
- `branch_k` (default: 2): Number of children per expansion
  - Higher values → more exploration but slower (more evaluations per iteration)
  - Trade-off between breadth and depth of search
- `expand_steps` (default: 20): Number of denoising steps during expansion
  - Higher values → children closer to terminal (faster reaching t=0)
  - Lower values → finer-grained search (more tree depth)

**Policy**: The action is sampled from the learned posterior: 
$$a_t \sim p_\theta(E_{t-1} | E_t, X, y, \text{spectrum})$$
where stochasticity comes from sampling discrete edges from categorical distributions.

**Output**: Updated tree, `[batch_size, K]` new child indices

### Step 5: Evaluation (`_batched_evaluate`)

**Purpose**: Score nodes using external verifier (ICEBERG model for MS/MS prediction)

**Input**: Tree, `[batch_size, K]` node indices, metadata, target spectra

**Process**:
1. **Gather node states** for evaluation: `(X_t, E_t, y_t, t_int)`

2. **Get molecular predictions**:
   - **If terminal** (`t_int == 0`): use current state directly
   - **If non-terminal** (`t_int > 0`): 
     - Predict clean molecule using diffusion model: `p_θ(E_0 | E_t, X, y)`
     - Sample from predicted distribution
     - Use predicted `E_0` for evaluation (X unchanged, edge-only)

3. **Convert to molecules**:
   - Convert graph tensors to RDKit molecules (not batchable, must loop)
   - Generate canonical SMILES for each molecule
   
4. **Batch score with deduplication**:
   - Deduplicate molecules by SMILES (avoid redundant ICEBERG calls)
   - Call `verifier.score_batch()` with ALL unique molecules at once:
     - Input: molecules, SMILES, precursor m/z, adducts, instruments, collision energies, target spectra
     - ICEBERG predicts MS/MS spectrum for each molecule: `F(mol) → predicted_spectrum`
     - Compute binned cosine similarity: 
       $$r = \text{CosineSimilarity}(\text{predicted\_spectrum}, \text{target\_spectrum})$$
   - Scatter scores back to original indices
   
5. **Store rewards in tree**: `tree.node_rewards[batch, node] = score`

**Reward Function**: 
$$r(s) = \frac{\sum_i \text{pred}_i \cdot \text{target}_i}{\sqrt{\sum_i \text{pred}_i^2} \cdot \sqrt{\sum_i \text{target}_i^2}}$$
where spectra are binned into m/z bins of size `bin_size`.

**Hyperparameters**:
- `similarity.bin_size` (default: 10.0): Bin size for spectrum binning (in Daltons)
  - Larger bins → coarser matching, more tolerance for m/z errors
  - Smaller bins → finer matching, stricter similarity
- `num_workers` (default: 64): Parallel workers for ICEBERG prediction
  - More workers → faster evaluation but more memory
- `verifier_batch_size` (default: 32): Internal batch size for verifier
  - Adjust based on GPU memory constraints

**Key Optimization**: Single batched verifier call with deduplication instead of sequential scoring

**Output**: `[batch_size, K]` similarity scores (range: [0, 1])

### Step 6: Backup (`_batched_backup`)

**Purpose**: Propagate evaluation scores up the tree to root nodes

**Input**: Tree, `[batch_size, K]` evaluated node indices, `[batch_size, K]` scores

**Process**:
1. **Flatten for processing**: Flatten to `[batch_size × K]` nodes and scores

2. **Iterative propagation** (up to `max_depth` iterations):
   - For each current node:
     - Update visit count: $N_{\text{new}}(s) = N_{\text{old}}(s) + 1$
     - Update cumulative value: $V_{\text{new}}(s) = V_{\text{old}}(s) + r$
     - Note: Value is **cumulative sum**, not average (Q-value computed as V/N during selection)
   - Move to parent nodes: `current_nodes ← parents[current_nodes]`
   - Stop when all paths reach root's parent (`NO_PARENT = -1`)

3. **Statistics tracked**:
   - `N(node)`: Number of times this node has been visited/evaluated
   - `V(node)`: Sum of all rewards backpropagated through this node
   - `r(node)`: This node's own immediate reward from evaluation
   - `Q(node) = V(node) / N(node)`: Average value (computed on-the-fly during selection)

**Formula**: For each node on the path from leaf to root:
$$N_{\text{new}}(s_j) = N_{\text{old}}(s_j) + 1$$
$$V_{\text{new}}(s_j) = V_{\text{old}}(s_j) + r_{\text{leaf}}$$

where $r_{\text{leaf}}$ is the reward from the evaluated leaf node.

**Output**: Updated tree with propagated statistics

### Step 7: Result Extraction (`_extract_batched_results`)

**Purpose**: Collect top-k molecules from completed MCTS trees

**Input**: Tree after `num_simulation_steps` iterations, target k molecules per sample

**Process**:

**Phase 1: Collect terminal nodes**
1. Identify all terminal nodes (`t_int == 0`) with visits > 0
2. Sort by reward (descending) per sample
3. For each sample, collect up to k terminal nodes:
   - Convert graph states to RDKit molecules and SMILES
   - Add to global molecule list

**Phase 2: Complete non-terminal nodes (if needed)**
1. If sample has < k molecules, select top non-terminal nodes by Q-value:
   - $Q(s) = V(s) / N(s)$
2. **Batched completion with dynamic masking**:
   - Group all nodes to complete across all samples
   - Process in mini-batches (size = `eval_batch_size`)
   - For each mini-batch:
     - Track current timestep per node
     - While any node has `t > 0`:
       - Create active mask: `active = (current_t > 0)`
       - Denoise active nodes: `p(E_{t-1} | E_t, X, y)`
       - Keep inactive nodes unchanged
       - Decrement timesteps: `t ← max(t-1, 0)`
   - Convert completed states to molecules

**Phase 3: Batch scoring**
1. Deduplicate all molecules by SMILES across entire batch
2. Single batched verifier call: score ALL unique molecules at once
3. Scatter scores back to per-sample results

**Phase 4: Return top-k per sample**
1. Sort molecules by score (descending) per sample
2. Return top k: `[(smiles, score, mol), ...]` for each sample

**Hyperparameters**:
- `test_samples_to_generate` (default: 100): Number of final molecules per sample
- `eval_batch_size` (default: 128): Mini-batch size for completing non-terminal nodes
  - Adjust based on GPU memory

**Key Optimization**: All molecule completion and scoring done in massive batches, not sequentially

**Output**: `List[List[(smiles, score, mol)]]` - one list per sample with top-k results

## Pseudocode

```python
def batched_mcts(spectra_batch, num_simulations, branch_k, c_puct, 
                 prediffuse_steps, expand_steps, test_k):
    """
    Batched MCTS for molecular generation from MS/MS spectra.
    
    Args:
        spectra_batch: [batch_size] MS/MS spectra
        num_simulations: Number of MCTS iterations
        branch_k: Number of children per expansion
        c_puct: UCT exploration constant
        prediffuse_steps: Pre-diffusion steps
        expand_steps: Denoising steps during expansion
        test_k: Number of molecules to return per sample
    """
    # Step 1: Initialize
    tree = initialize_batched_tree(spectra_batch)
    # tree.node_states: [batch_size, max_nodes, n_atoms, ...]
    # tree.node_visits, node_values, node_rewards: [batch_size, max_nodes]
    
    # Step 2: Pre-diffuse from T to t_thresh
    t_thresh = T - prediffuse_steps
    for t in range(T-1, t_thresh-1, -1):
        tree.node_states_E[:, 0] = denoise(tree.node_states_E[:, 0], t)
    tree.node_timesteps_int[:, 0] = t_thresh
    
    # Step 3-6: Main MCTS loop
    for simulation in range(num_simulations):
        # Selection: traverse from root using UCT
        leaf_indices = []  # [batch_size]
        for b in range(batch_size):
            node = ROOT
            while has_children(tree, b, node):
                # Compute UCT for all children
                uct = Q(children) + c_puct * sqrt(log(N(node)) / N(children))
                node = argmax(uct)
            leaf_indices.append(node)
        
        # Expansion: create branch_k children per leaf
        new_children = []  # [batch_size, branch_k]
        for step in range(expand_steps):
            leaf_states = gather(tree, leaf_indices)  # [batch_size, ...]
            repeated = repeat(leaf_states, branch_k)  # [batch_size*K, ...]
            
            # Denoise with dynamic masking
            active = (current_t > 0)
            denoised = where(active, denoise(repeated), repeated)
            current_t = where(active, current_t - 1, current_t)
        
        # Store children
        for b, k in product(batch_size, branch_k):
            child_idx = allocate_node(tree, b)
            tree.node_states[b, child_idx] = denoised[b*K + k]
            new_children[b, k] = child_idx
        
        # Evaluation: batch score all children
        all_mols = [to_molecule(tree.node_states[b, c]) 
                    for b in range(batch_size) for c in new_children[b]]
        scores = verifier.score_batch(all_mols)  # Single batched call
        scores = reshape(scores, [batch_size, branch_k])
        
        # Backup: propagate to root
        for b, k in product(batch_size, branch_k):
            node = new_children[b, k]
            reward = scores[b, k]
            while node != NO_PARENT:
                tree.node_visits[b, node] += 1
                tree.node_values[b, node] += reward
                node = tree.parents[b, node]
    
    # Step 7: Extract results
    results = []
    for b in range(batch_size):
        # Collect terminals
        terminals = [(node, tree.node_rewards[b, node]) 
                     for node in range(tree.num_nodes[b])
                     if tree.is_terminal[b, node] and tree.node_visits[b, node] > 0]
        terminals = sorted(terminals, key=lambda x: x[1], reverse=True)
        
        mols = [to_molecule(tree.node_states[b, n]) for n, _ in terminals[:test_k]]
        
        # Complete non-terminals if needed
        if len(mols) < test_k:
            non_terminals = [(node, Q(b, node)) 
                            for node in range(tree.num_nodes[b])
                            if not tree.is_terminal[b, node]]
            non_terminals = sorted(non_terminals, key=lambda x: x[1], reverse=True)
            
            for node, _ in non_terminals[:test_k - len(mols)]:
                completed = complete_to_terminal(tree.node_states[b, node])
                mols.append(completed)
        
        # Batch score
        scores = verifier.score_batch(mols)
        results.append(sorted(zip(mols, scores), key=lambda x: x[1], reverse=True)[:test_k])
    
    return results
```

## Summary of Hyperparameters

All hyperparameters from `run_mcts.sh` and their effects:

### Core MCTS Parameters
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `mcts.num_simulation_steps` | 3000 | 100-10000 | Number of MCTS iterations. Higher → more thorough search but slower |
| `mcts.branch_k` | 2 | 2-20 | Children per expansion. Higher → more exploration per iteration but slower |
| `mcts.c_puct` | 0.25 | 0.0-2.0 | UCT exploration constant. Higher → favor unexplored nodes, Lower → favor high-value nodes |

### Diffusion Control
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `mcts.prediffuse_steps` | 200 | 0-500 | Pre-diffusion steps from T. Higher → start search closer to clean molecules |
| `mcts.expand_steps` | 20 | 1-100 | Denoising steps during expansion. Higher → children closer to terminal |
| `model.diffusion_steps` | 500 | - | Total diffusion timesteps T (set by pre-trained model) |

### Verifier (ICEBERG) Parameters
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `mcts.num_workers` | 64 | 1-128 | Parallel workers for ICEBERG. More → faster but more memory |
| `mcts.verifier_batch_size` | 32 | 8-256 | Internal batch size for verifier. Adjust based on GPU memory |
| `mcts.similarity.bin_size` | 10.0 | 0.1-50.0 | Spectrum binning size (Da). Larger → more tolerance for m/z errors |
| `mcts.verifier_type` | 'iceberg' | - | Verifier model type |

### Evaluation Parameters
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `general.test_samples_to_generate` | 100 | 1-1000 | Number of molecules to return per spectrum |
| `train.eval_batch_size` | 128 | 16-512 | Mini-batch size for completing non-terminal nodes |
| `dataset.max_count` | 2000 | - | Maximum number of test samples (for debugging) |

### Other Parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `general.gpus` | 1 | Number of GPUs to use |
| `general.seed` | 123 | Random seed for reproducibility |
| `mcts.use_mcts` | true | Enable MCTS (vs. standard diffusion sampling) |
| `mcts.time_budget_s` | 0.0 | Time budget per sample (0 = no limit) |

### Recommended Configurations

**Fast exploration (debugging)**:
```bash
mcts.num_simulation_steps=500
mcts.branch_k=2
mcts.expand_steps=50
mcts.prediffuse_steps=300
```

**Balanced (default)**:
```bash
mcts.num_simulation_steps=3000
mcts.branch_k=2
mcts.expand_steps=20
mcts.prediffuse_steps=200
```

**Thorough search (slow)**:
```bash
mcts.num_simulation_steps=10000
mcts.branch_k=5
mcts.expand_steps=10
mcts.prediffuse_steps=100
```

**High diversity**:
```bash
mcts.c_puct=1.0  # More exploration
mcts.branch_k=10
mcts.expand_steps=5  # Finer-grained search
```


# Implementation Details & Design Decisions

## Handling Edge Cases

### 1. Terminal nodes selected for expansion
**Solution**: Terminal nodes evaluate themselves K times (creating K copies with same reward) for batch structure consistency. The backup phase propagates their rewards normally.

### 2. Already-expanded nodes selected for expansion
**Solution**: Selection phase avoids this by only selecting leaves (nodes without children) or terminals. The `_batched_select` traverses until finding unexpanded nodes.

### 3. Insufficient terminal nodes at extraction
**Solution**: Extract top Q-valued non-terminal nodes and complete them to t=0 using batched dynamic masking. All completions happen in parallel, then a single batch scoring call evaluates all molecules.

### 4. Duplicate molecules during expansion
**Solution**: Duplicates are handled naturally by the tree structure (each gets its own node with independent statistics). During final evaluation, deduplication happens before verifier call to avoid redundant ICEBERG predictions, and scores are scattered back to all duplicates.

### 5. Invalid molecules from graph-to-RDKit conversion
**Solution**: Invalid molecules receive a score of -1.0. These typically don't propagate well in the tree due to low rewards.

## Key Optimizations

### Batching Strategy
1. **Tree operations**: Fully batched across samples using tensors `[batch_size, max_nodes, ...]`
2. **Expansion**: All K children denoised simultaneously via `[batch_size × K, ...]` tensors
3. **Evaluation**: Single verifier call for all children with SMILES-based deduplication
4. **Result extraction**: All molecule completions and final scoring in massive batches

### Memory Efficiency
1. **Uint8 storage**: Node states stored as uint8 indices (not one-hot) to save memory
2. **Dynamic masking**: Simultaneous denoising with per-node timestep tracking avoids redundant operations
3. **Shared cache**: Multiprocessing workers share spectrum prediction cache via Manager.dict()

### Computational Efficiency
1. **Worker pools**: Persistent multiprocessing pool (64 workers default) for ICEBERG predictions
2. **Cache hits**: Both spectrum and score caches reduce redundant ICEBERG calls
3. **Chunked processing**: Non-terminal completion uses mini-batches to balance memory and speed

# Dataset

| Model         | Dataset                | Specials                                                            |
| ------------- | ---------------------- | ------------------------------------------------------------------- |
| DiffMS        | NPLIB1, MassSpecGym    | Adduct is needed, energy does not matter                            |
| ICEBERG (old) | NPLIB1, NIST20         | Adduct is needed, energy does not matter                            |
| ICEBERG (new) | NIST20                 | Both adduct and energy are needed                                   |
| MARASON       | NIST20, MassSpecGym    | Both adduct and energy are needed                                   |
| SIRIUS        | MassBank, GNPS, NIST17 | Takes in the spectra and adduct info, returns the chemical formulae |
| MIST          | --                     | The pretrained version does not matter                              |

## Model Architecture

The implementation combines three key components:

### 1. Spectral Encoder (SpectraEncoderGrowing from MIST)
- **Input**: MS/MS peaks `(m/z, intensity)` pairs + metadata (precursor m/z, adduct)
- **Architecture**: Transformer-based with peak attention and formula embeddings
- **Output**: Fixed-size embedding (4096-D) that conditions the diffusion model
- **Purpose**: Extract spectral features for molecular generation

### 2. Graph Diffusion Model (GraphTransformer from DiffMS)
- **Input**: Noisy molecular graphs + spectral embedding + timestep
- **Architecture**: Graph transformer operating on dense adjacency matrices
- **Output**: Denoised predictions for edges (bonds) at previous timestep
- **Purpose**: Iteratively denoise from noise to clean molecular structures
- **Key property**: Edge-only denoising (atom types X fixed, only bonds E denoised)

### 3. Verifier (ICEBERG)
- **Input**: Molecular structures (SMILES) + metadata (adduct, collision energy, instrument)
- **Architecture**: Graph neural network + fragment tree for MS/MS prediction
- **Output**: Predicted MS/MS spectrum for input molecule
- **Purpose**: Evaluate molecule quality by comparing predicted vs. target spectra
- **Implementation**: Multiprocessing pool with 64 workers + shared caching

### Integration Flow
```
Target Spectrum → [Encoder] → Spectral Embedding
                                     ↓
Noisy Graph (t=T) → [Diffusion + MCTS] → Clean Graph (t=0)
                          ↑
                    [Verifier] scores guide search
```

## Key Differences from Standard MCTS

### 1. Continuous State Space with Timesteps
Unlike traditional MCTS (e.g., AlphaGo) with discrete board states, each node has:
- Molecular graph state `(X, E, y)`
- Diffusion timestep `t ∈ [0, T]`
- Nodes at different timesteps represent different levels of "completion"

### 2. Stochastic Actions from Learned Policy
- Actions are not deterministic (like chess moves)
- Each child samples from learned posterior: `p_θ(E_{t-1} | E_t, X, y, spectrum)`
- K children from same parent can be different due to sampling stochasticity

### 3. Multi-Step Expansion
- Standard MCTS: expand by 1 action
- This implementation: expand by `expand_steps` denoising steps
- Trade-off: coarser search tree but faster convergence to terminals

### 4. External Verifier for Evaluation
- Standard MCTS: immediate reward from environment
- This implementation: ICEBERG model predicts spectrum and computes similarity
- Expensive evaluation (neural network forward pass) motivates caching and batching

### 5. Pre-diffusion Strategy
- Start search from `t = T - prediffuse_steps` instead of `t = T`
- Reduces search space by "warming up" to intermediate noise level
- Similar to warm-starting in optimization

### 6. Edge-Only Denoising
- Atom types (X) fixed throughout search
- Only bond types (E) are denoised
- Reduces action space and speeds up sampling

### 7. Batched Operations
- Process multiple samples simultaneously
- Amortize neural network and verifier costs
- Key to practical runtime efficiency

## Future Extensions

### Potential Improvements
1. **Adaptive expand_steps**: Start with large steps, decrease as approaching terminal
2. **Progressive widening**: Start with small branch_k, increase over time
3. **Value network**: Train a neural network to predict Q-values (like AlphaZero)
4. **Policy guidance**: Use diffusion model's predicted probabilities for prior in UCT
5. **Hierarchical search**: Coarse-to-fine approach with different bin_sizes

### Extension to Other Generative Models
The MCTS framework can be adapted to:
- **Spec2Mol** (autoregressive): MCTS over token sequences
- **MoLFormer** (transformer): MCTS over latent representations
- **Other diffusion models**: Replace DiffMS with alternative architectures

### Scalability
Current bottlenecks and solutions:
- **Memory**: Tree grows O(M × K) nodes. Solution: tree pruning, node recycling
- **Verifier speed**: ICEBERG is expensive. Solution: distilled verifier, approximate scoring
- **Denoising speed**: Many forward passes. Solution: cached model states, knowledge distillation
