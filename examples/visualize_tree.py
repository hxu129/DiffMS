#!/usr/bin/env python3
"""
Visualize MCTS search tree structure.

Usage:
    python visualize_tree.py path/to/mcts_tree_structure_*.json

Creates interactive and static visualizations of the MCTS search tree showing:
- Tree topology (parent-child relationships)
- Node properties (visits, Q-values, rewards, timesteps)
- Terminal vs non-terminal nodes
- Most visited paths
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import deque


def load_tree_data(filepath):
    """Load tree structure from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_node_positions_hierarchical(nodes, edges):
    """
    Compute node positions using hierarchical layout (depth-based).
    
    Returns:
        pos: dict mapping node_id -> (x, y) position
    """
    # Build adjacency list
    children = {node['id']: [] for node in nodes}
    parent_map = {}
    
    for edge in edges:
        children[edge['source']].append(edge['target'])
        parent_map[edge['target']] = edge['source']
    
    # Find root (node with no parent)
    root = None
    for node in nodes:
        if node['id'] not in parent_map:
            root = node['id']
            break
    
    if root is None:
        # Fallback: use node 0 as root
        root = 0
    
    # BFS to assign levels
    node_levels = {}
    queue = deque([(root, 0)])
    max_level = 0
    
    while queue:
        node_id, level = queue.popleft()
        node_levels[node_id] = level
        max_level = max(max_level, level)
        
        for child in children.get(node_id, []):
            queue.append((child, level + 1))
    
    # Count nodes per level
    level_counts = {}
    for node_id, level in node_levels.items():
        level_counts[level] = level_counts.get(level, 0) + 1
    
    # Assign positions
    level_indices = {level: 0 for level in range(max_level + 1)}
    pos = {}
    
    # Use BFS again to assign x positions left-to-right
    queue = deque([root])
    visited = set([root])
    
    while queue:
        node_id = queue.popleft()
        level = node_levels[node_id]
        
        # x position: spread evenly across level width
        level_width = level_counts[level]
        x = (level_indices[level] + 0.5) / level_width if level_width > 0 else 0.5
        level_indices[level] += 1
        
        # y position: based on depth (inverted so root is at top)
        y = max_level - level
        
        pos[node_id] = (x, y)
        
        # Add children to queue
        for child in sorted(children.get(node_id, [])):
            if child not in visited:
                queue.append(child)
                visited.add(child)
    
    return pos


def visualize_tree(tree_data, output_dir, filter_visits=5):
    """
    Create comprehensive tree visualizations.
    
    Args:
        tree_data: Loaded tree structure data
        output_dir: Directory to save plots
    """
    nodes = tree_data['nodes']
    edges = tree_data['edges']
    config = tree_data['config']
    
    if not nodes:
        print("No nodes to visualize!")
        return
    
    # Filter nodes: only show nodes with visit_count > 3 OR terminal nodes
    filtered_nodes = []
    node_id_mapping = {}  # Map old node IDs to new indices
    for idx, node in enumerate(nodes):
        if node['visits'] > filter_visits or node['is_terminal']:
            new_idx = len(filtered_nodes)
            node_id_mapping[node['id']] = new_idx
            filtered_nodes.append(node)
    
    print(f"Filtering nodes: {len(nodes)} total -> {len(filtered_nodes)} displayed "
          f"(visit_count > {filter_visits} or terminal)")
    
    # Filter edges: only keep edges where both source and target are in filtered_nodes
    filtered_edges = []
    filtered_node_ids = {node['id'] for node in filtered_nodes}
    for edge in edges:
        if edge['source'] in filtered_node_ids and edge['target'] in filtered_node_ids:
            filtered_edges.append(edge)
    
    if not filtered_nodes:
        print(f"No nodes match the filter criteria (visit_count > {filter_visits} or terminal)!")
        return
    
    # Use filtered nodes and edges for visualization
    nodes = filtered_nodes
    edges = filtered_edges
    
    # Compute layout with filtered nodes
    pos = compute_node_positions_hierarchical(nodes, edges)
    
    # Create node lookup
    node_dict = {n['id']: n for n in nodes}
    
    # Extract properties for visualization
    node_ids = [n['id'] for n in nodes]
    visits = np.array([n['visits'] for n in nodes])
    q_values = np.array([n['q_value'] for n in nodes])
    rewards = np.array([n['reward'] for n in nodes])
    is_terminal = np.array([n['is_terminal'] for n in nodes])
    timesteps = np.array([n['timestep'] for n in nodes])
    
    # Normalize for coloring
    visits_norm = visits / visits.max() if visits.max() > 0 else visits
    
    # Use non-linear scaling for node sizes to make differences more visible
    # Square root scaling makes size differences more dramatic
    # High-visit nodes will be much larger, low-visit nodes much smaller
    visits_norm_sized = np.sqrt(visits_norm)  # Square root scaling for more contrast
    
    # Helper function to draw edges
    def draw_edges(ax, pos, edges, linewidth=2.0, alpha=0.3, color='k'):
        for edge in edges:
            src, tgt = edge['source'], edge['target']
            if src in pos and tgt in pos:
                x = [pos[src][0], pos[tgt][0]]
                y = [pos[src][1], pos[tgt][1]]
                ax.plot(x, y, color, alpha=alpha, linewidth=linewidth)
    
    # Calculate node sizes once
    node_sizes = np.array([50 + 950 * visits_norm_sized[i] for i in range(len(node_ids))])
    
    # ========== Plot 1: Visit Count Coloring ==========
    fig1, ax1 = plt.subplots(1, 1, figsize=(40, 24))
    ax1.set_title('MCTS Tree: Visit Counts', fontsize=56, fontweight='bold', pad=20)
    
    draw_edges(ax1, pos, edges, linewidth=2.0, alpha=0.3)
    
    # Draw nodes
    for i, node_id in enumerate(node_ids):
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        color = 'lightgreen' if is_terminal[i] else 'lightblue'
        ax1.scatter([x], [y], s=node_sizes[i], c=[color], 
                   edgecolors='black', linewidths=2.0, alpha=0.7, zorder=3)
        # Label top nodes
        if visits[i] >= sorted(visits, reverse=True)[min(10, len(visits)-1)]:
            ax1.text(x, y, str(node_id), ha='center', va='center', 
                    fontsize=24, fontweight='bold')
    
    ax1.axis('off')
    ax1.legend(handles=[
        mpatches.Patch(color='lightblue', label='Non-terminal'),
        mpatches.Patch(color='lightgreen', label='Terminal')
    ], loc='upper right', fontsize=28)
    
    plt.tight_layout()
    output_path = output_dir / 'tree_visit_counts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # ========== Plot 2: Q-Value Coloring ==========
    fig2, ax2 = plt.subplots(1, 1, figsize=(40, 24))
    ax2.set_title('MCTS Tree: Q-Values', fontsize=56, fontweight='bold', pad=20)
    
    draw_edges(ax2, pos, edges, linewidth=2.0, alpha=0.3)
    
    # Draw nodes colored by Q-value
    scatter = None
    for i, node_id in enumerate(node_ids):
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        scatter = ax2.scatter([x], [y], s=node_sizes[i], c=[q_values[i]], 
                             cmap='RdYlGn', vmin=q_values.min(), vmax=q_values.max(),
                             edgecolors='black', linewidths=2.0, alpha=0.8, zorder=3)
    
    if scatter:
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Q-value', fontsize=28)
        cbar.ax.tick_params(labelsize=20)
    ax2.axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'tree_q_values.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # ========== Plot 3: Reward Coloring ==========
    fig3, ax3 = plt.subplots(1, 1, figsize=(40, 24))
    ax3.set_title('MCTS Tree: Rewards', fontsize=56, fontweight='bold', pad=20)
    
    draw_edges(ax3, pos, edges, linewidth=2.0, alpha=0.3)
    
    # Draw nodes colored by reward
    scatter = None
    for i, node_id in enumerate(node_ids):
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        scatter = ax3.scatter([x], [y], s=node_sizes[i], c=[rewards[i]], 
                             cmap='plasma', vmin=max(0, rewards.min()), vmax=rewards.max(),
                             edgecolors='black', linewidths=2.0, alpha=0.8, zorder=3)
    
    if scatter:
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Reward', fontsize=28)
        cbar.ax.tick_params(labelsize=20)
    ax3.axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'tree_rewards.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # ========== Plot 4: Timestep Progression ==========
    fig4, ax4 = plt.subplots(1, 1, figsize=(40, 24))
    ax4.set_title('MCTS Tree: Timestep Progression', fontsize=56, fontweight='bold', pad=20)
    
    draw_edges(ax4, pos, edges, linewidth=2.0, alpha=0.3)
    
    # Draw nodes colored by timestep
    scatter = None
    for i, node_id in enumerate(node_ids):
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        scatter = ax4.scatter([x], [y], s=node_sizes[i], c=[timesteps[i]], 
                             cmap='viridis_r', vmin=0, vmax=timesteps.max(),
                             edgecolors='black', linewidths=2.0, alpha=0.8, zorder=3)
    
    if scatter:
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Timestep t', fontsize=28)
        cbar.ax.tick_params(labelsize=20)
    ax4.axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'tree_timesteps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # ========== Plot 5: Top Paths Highlighted ==========
    fig5, ax5 = plt.subplots(1, 1, figsize=(40, 24))
    ax5.set_title('MCTS Tree: Top-5 Most Visited Paths', fontsize=56, fontweight='bold', pad=20)
    
    # Find top-5 most visited terminal nodes
    terminal_nodes = [(i, node_id) for i, node_id in enumerate(node_ids) if is_terminal[i]]
    terminal_nodes_sorted = sorted(terminal_nodes, key=lambda x: visits[x[0]], reverse=True)
    top_terminal = terminal_nodes_sorted[:5]
    
    # Trace paths to root
    top_paths = []
    for idx, node_id in top_terminal:
        path = [node_id]
        current = node_id
        while current in [e['target'] for e in edges]:
            parent = next((e['source'] for e in edges if e['target'] == current), None)
            if parent is None:
                break
            path.append(parent)
            current = parent
        top_paths.append(path[::-1])  # Reverse to root->leaf
    
    # Draw all edges in gray
    draw_edges(ax5, pos, edges, linewidth=2.0, alpha=0.2, color='gray')
    
    # Highlight top paths
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    for path_idx, path in enumerate(top_paths):
        color = colors[path_idx % len(colors)]
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i+1]
            if src in pos and tgt in pos:
                x = [pos[src][0], pos[tgt][0]]
                y = [pos[src][1], pos[tgt][1]]
                ax5.plot(x, y, color, alpha=0.8, linewidth=8, zorder=2)
    
    # Draw nodes
    for i, node_id in enumerate(node_ids):
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        color = 'lightgreen' if is_terminal[i] else 'lightblue'
        ax5.scatter([x], [y], s=node_sizes[i], c=[color], 
                   edgecolors='black', linewidths=2.0, alpha=0.7, zorder=3)
    
    ax5.axis('off')
    ax5.legend(handles=[
        mpatches.Patch(color='red', label='Rank 1 path'),
        mpatches.Patch(color='orange', label='Rank 2 path'),
        mpatches.Patch(color='yellow', label='Rank 3 path'),
        mpatches.Patch(color='green', label='Rank 4 path'),
        mpatches.Patch(color='blue', label='Rank 5 path'),
    ], loc='upper right', fontsize=32)
    
    plt.tight_layout()
    output_path = output_dir / 'tree_top_paths.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # ========== Plot 6: Statistics ==========
    fig6, ax6 = plt.subplots(1, 1, figsize=(20, 12))
    ax6.axis('off')
    
    # Compute statistics
    stats_text = f"""
MCTS Tree Statistics
{'='*40}

Total Nodes: {len(nodes)}
Terminal Nodes: {is_terminal.sum()}
Non-terminal Nodes: {(~is_terminal).sum()}

Visit Statistics:
  Mean: {visits.mean():.1f}
  Median: {np.median(visits):.1f}
  Max: {visits.max()}
  
Q-Value Statistics:
  Mean: {q_values.mean():.4f}
  Std: {q_values.std():.4f}
  Min: {q_values.min():.4f}
  Max: {q_values.max():.4f}

Reward Statistics:
  Mean: {rewards.mean():.4f}
  Std: {rewards.std():.4f}
  Min: {rewards.min():.4f}
  Max: {rewards.max():.4f}

MCTS Configuration:
  branch_k: {config['branch_k']}
  c_puct: {config['c_puct']}
  expand_steps: {config['expand_steps']}
  prediffuse_steps: {config['prediffuse_steps']}

Top-5 Most Visited Nodes:
"""
    
    top_5_nodes = sorted(range(len(visits)), key=lambda i: visits[i], reverse=True)[:5]
    for rank, idx in enumerate(top_5_nodes, 1):
        node_id = node_ids[idx]
        stats_text += f"  {rank}. Node {node_id}: visits={visits[idx]}, Q={q_values[idx]:.4f}, t={timesteps[idx]}\n"
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=36, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / 'tree_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    print(f"\n✅ All tree visualizations saved to: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_tree.py <path_to_tree_structure.json> [filter_visits]")
        sys.exit(1)
    
    tree_file = Path(sys.argv[1])
    if not tree_file.exists():
        print(f"Error: File not found: {tree_file}")
        sys.exit(1)
    
    print(f"Loading tree structure from: {tree_file}")
    tree_data = load_tree_data(tree_file)
    
    # Create output directory
    output_dir = tree_file.parent / 'tree_visualizations'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    filter_visits = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Generate visualizations
    print("\nGenerating tree visualizations...")
    visualize_tree(tree_data, output_dir, filter_visits=filter_visits)
    
    print(f"\n✅ Visualization complete!")
    print(f"\nTree statistics:")
    print(f"  Sample: {tree_data['sample_idx']}")
    print(f"  Total nodes: {tree_data['num_nodes']}")
    print(f"  Visited nodes: {tree_data['num_visited']}")
    print(f"  Edges: {len(tree_data['edges'])}")


if __name__ == '__main__':
    main()

