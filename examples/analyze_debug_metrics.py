#!/usr/bin/env python3
"""
Analyze MCTS debug metrics from saved pickle files.

Usage:
    python analyze_debug_metrics.py path/to/mcts_debug_metrics.pkl

This script loads debug metrics and creates visualizations to help
understand MCTS behavior.
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics(filepath):
    """Load debug metrics from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_tree_growth(metrics, output_dir):
    """Plot tree size over simulation steps."""
    steps = metrics.get('simulation_step_markers', [])
    sizes = metrics.get('tree_size_history', [])
    
    if not steps or not sizes:
        print("No tree growth data available")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, sizes, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Tree Size (Number of Nodes)', fontsize=12)
    plt.title('MCTS Tree Growth Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'tree_growth.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_reward_distribution(metrics, output_dir):
    """Plot histogram of reward distribution."""
    rewards = metrics.get('reward_history', [])
    
    if not rewards:
        print("No reward data available")
        return
    
    rewards = np.array(rewards)
    
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Reward Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('MCTS Reward Distribution', fontsize=14, fontweight='bold')
    
    # Add statistics
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    plt.axvline(mean_reward, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.3f}')
    plt.axvline(median_reward, color='g', linestyle='--', linewidth=2, label=f'Median: {median_reward:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = output_dir / 'reward_distribution.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print(f"\nReward Statistics:")
    print(f"  Mean:   {np.mean(rewards):.4f}")
    print(f"  Median: {np.median(rewards):.4f}")
    print(f"  Std:    {np.std(rewards):.4f}")
    print(f"  Min:    {np.min(rewards):.4f}")
    print(f"  Max:    {np.max(rewards):.4f}")
    
    # Sparsity analysis
    near_zero = np.sum(np.abs(rewards) < 0.01)
    sparsity = near_zero / len(rewards)
    print(f"  Sparsity (|r| < 0.01): {sparsity:.2%}")


def plot_q_value_trends(metrics, output_dir):
    """Plot Q-value trends for top nodes."""
    q_history = metrics.get('q_value_history', [])
    
    if not q_history:
        print("No Q-value history available")
        return
    
    # Extract top node Q-values over time
    steps = [entry['step'] for entry in q_history]
    top_q_values = [entry['top_Q'][0] if entry['top_Q'] else 0.0 for entry in q_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, top_q_values, marker='o', linewidth=2, markersize=4, color='purple')
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Q-Value (Top Node)', fontsize=12)
    plt.title('Q-Value Trend for Most-Visited Node', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'q_value_trend.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_visit_balance(metrics, output_dir):
    """Plot visit balance for root children over time."""
    visits_history = metrics.get('root_children_visits', [])
    steps = metrics.get('simulation_step_markers', [])
    
    if not visits_history or not steps:
        print("No visit balance data available")
        return
    
    # Calculate balance ratio over time
    balance_ratios = []
    for visits in visits_history:
        if visits and max(visits) > 0:
            balance_ratios.append(min(visits) / max(visits))
        else:
            balance_ratios.append(0.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, balance_ratios, marker='o', linewidth=2, markersize=4, color='orange')
    plt.axhline(y=0.1, color='r', linestyle='--', linewidth=1, label='Warning threshold (0.1)')
    plt.axhline(y=0.3, color='g', linestyle='--', linewidth=1, label='Good threshold (0.3)')
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Visit Balance Ratio (min/max)', fontsize=12)
    plt.title('Root Children Visit Balance Over Time', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'visit_balance.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print final balance
    if balance_ratios:
        final_balance = balance_ratios[-1]
        print(f"\nFinal Visit Balance Ratio: {final_balance:.3f}")
        if final_balance < 0.1:
            print("  ⚠️  WARNING: Highly imbalanced! Consider increasing c_puct.")
        elif final_balance < 0.3:
            print("  ⚡ Moderate imbalance. May want to increase c_puct slightly.")
        else:
            print("  ✅ Good exploration balance!")


def print_summary(metrics):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("MCTS DEBUG METRICS SUMMARY")
    print("="*80)
    
    # Tree growth
    sizes = metrics.get('tree_size_history', [])
    if sizes:
        print(f"\nTree Growth:")
        print(f"  Initial size: {sizes[0]}")
        print(f"  Final size:   {sizes[-1]}")
        if len(sizes) > 1:
            growth_rate = (sizes[-1] - sizes[0]) / len(sizes)
            print(f"  Growth rate:  {growth_rate:.2f} nodes/checkpoint")
    
    # Visit balance
    visits_history = metrics.get('root_children_visits', [])
    if visits_history and visits_history[-1]:
        final_visits = visits_history[-1]
        print(f"\nFinal Root Children Visits:")
        for i, v in enumerate(final_visits):
            print(f"  Child {i}: {v} visits")
    
    print("\n" + "="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_debug_metrics.py <path_to_metrics.pkl>")
        print("\nTo save metrics during MCTS, add this to your code:")
        print("  if model.mcts_config['debug_logging']:")
        print("      import pickle")
        print("      with open('mcts_debug_metrics.pkl', 'wb') as f:")
        print("          pickle.dump(model.debug_metrics, f)")
        sys.exit(1)
    
    metrics_path = Path(sys.argv[1])
    if not metrics_path.exists():
        print(f"Error: File not found: {metrics_path}")
        sys.exit(1)
    
    print(f"Loading metrics from: {metrics_path}")
    metrics = load_metrics(metrics_path)
    
    # Create output directory
    output_dir = metrics_path.parent / 'debug_plots'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_tree_growth(metrics, output_dir)
    plot_reward_distribution(metrics, output_dir)
    plot_q_value_trends(metrics, output_dir)
    plot_visit_balance(metrics, output_dir)
    
    # Print summary
    print_summary(metrics)
    
    print(f"\n✅ Analysis complete! Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

