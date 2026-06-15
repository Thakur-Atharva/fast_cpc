"""
Base classes and shared utilities for causal discovery runners.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class DiscoveryResult:
    """Standardized output from any causal discovery runner."""
    algorithm_name: str
    edges: List[Tuple[str, str, dict]]  # (source, target, metadata)
    elapsed_seconds: float
    adj_matrix: Optional[np.ndarray] = None
    graph_score: Optional[float] = None  # GES total score
    params_summary: str = ""  # Key params for title/filename


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def build_timeline_map(stage_cols: List[str]) -> Dict[str, int]:
    """Extract the leading integer from each column name for chronological ordering."""
    def _get_stage_num(col: str) -> int:
        match = re.match(r'^\d+', col)
        return int(match.group()) if match else 0
    return {col: _get_stage_num(col) for col in stage_cols}


def auto_select_trusted_hubs(
    df: pd.DataFrame,
    stage_cols: List[str],
    min_frequency: float = 0.15,
    max_frequency: float = 0.85,
    max_choice: int = 30,
) -> List[str]:
    """
    Identify high-signal hub columns based on data density and entropy,
    avoiding near-uniform columns that cause spurious correlations.
    """
    total_units = len(df)
    candidate_hubs = []

    for col in stage_cols:
        presence_ratio = df[col].sum() / total_units
        if min_frequency <= presence_ratio <= max_frequency:
            candidate_hubs.append((col, presence_ratio))

    # Sort by proximity to 50/50 split (maximum entropy)
    candidate_hubs.sort(key=lambda x: abs(x[1] - 0.5))
    final_hubs = [hub[0] for hub in candidate_hubs[:max_choice]]

    print(f"  Hub analysis: {len(candidate_hubs)} candidates, selected top {len(final_hubs)}")
    for hub, ratio in candidate_hubs[:5]:
        print(f"    -> {hub:<20s} | Presence: {ratio*100:.1f}%")
    if len(final_hubs) > 5:
        print(f"    ... and {len(final_hubs) - 5} more")

    return final_hubs


def apply_temporal_filter(
    G: nx.DiGraph,
    timeline_map: Dict[str, int],
) -> nx.DiGraph:
    """Remove edges that violate chronological ordering (src_time >= dest_time)."""
    edges_to_remove = []
    for src, dest in G.edges():
        if timeline_map.get(src, 0) >= timeline_map.get(dest, 0):
            edges_to_remove.append((src, dest))
    G.remove_edges_from(edges_to_remove)
    return G


def apply_transitive_reduction(G: nx.DiGraph) -> nx.DiGraph:
    """Apply transitive reduction to remove structurally redundant edges."""
    try:
        G_reduced = nx.transitive_reduction(G)
        G_reduced.add_nodes_from(G.nodes(data=True))
        return G_reduced
    except Exception as e:
        print(f"  Warning: transitive reduction bypassed ({e})")
        return G


def print_results(result: DiscoveryResult):
    """Print a standardized summary of discovered edges."""
    print(f"\n{'=' * 70}")
    print(f"  DISCOVERED CAUSAL GRAPH SUMMARY ({result.algorithm_name})")
    print(f"{'=' * 70}")
    print(f"  Algorithm Time : {result.elapsed_seconds:.2f}s")
    print(f"  Total Edges    : {len(result.edges)}")
    if result.graph_score is not None:
        print(f"  Graph Score    : {result.graph_score:.4f}")
    print(f"  Parameters     : {result.params_summary}")
    print()

    if not result.edges:
        print("  (No edges discovered)")
        return

    # Determine column widths
    src_w = max(len(e[0]) for e in result.edges)
    tgt_w = max(len(e[1]) for e in result.edges)
    src_w = max(src_w, 6)
    tgt_w = max(tgt_w, 6)

    header = f"  {'#':>3} | {'Source':<{src_w}} | {'Target':<{tgt_w}} | {'Type':<20} | Score/Info"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for i, (src, tgt, meta) in enumerate(result.edges, 1):
        edge_type = meta.get("edge_type", "directed")
        score_str = ""
        if "score" in meta and meta["score"] is not None:
            score_str = f"{meta['score']:.4f}"
        elif "p_value" in meta and meta["p_value"] is not None:
            score_str = f"p={meta['p_value']:.2e}"
        else:
            score_str = "—"
        print(f"  {i:>3} | {src:<{src_w}} | {tgt:<{tgt_w}} | {edge_type:<20} | {score_str}")

    print()


def save_visualization(
    G: nx.DiGraph,
    stage_cols: List[str],
    timeline_map: Dict[str, int],
    result: DiscoveryResult,
    output_path: Optional[str] = None,
):
    """Render and save a chronological DAG visualization."""
    if output_path is None:
        output_path = f"{result.algorithm_name.lower()}_manufacturing_dag.png"

    plt.figure(figsize=(18, 6))
    title = f"Causal Graph — {result.algorithm_name}"
    if result.params_summary:
        title += f" ({result.params_summary})"
    plt.title(title, fontsize=14, fontweight='bold', pad=15)

    # Layout nodes chronologically
    sorted_stages = sorted(stage_cols, key=lambda x: timeline_map.get(x, 0))
    pos = {}
    for idx, node in enumerate(sorted_stages):
        pos[node] = (timeline_map.get(node, 0) * 3, (idx % 2) * 2 - 1)

    # Only draw nodes that have edges
    nodes_with_edges = set()
    for src, tgt in G.edges():
        nodes_with_edges.add(src)
        nodes_with_edges.add(tgt)

    if not nodes_with_edges:
        print(f"  No edges to visualize, skipping PNG.")
        plt.close()
        return

    subgraph = G.subgraph(nodes_with_edges)
    sub_pos = {n: pos[n] for n in nodes_with_edges if n in pos}

    nx.draw_networkx_nodes(subgraph, sub_pos, node_color='lightgreen',
                           node_size=700, edgecolors='black', linewidths=1.0)
    nx.draw_networkx_edges(subgraph, sub_pos, arrows=True, arrowstyle='-|>',
                           arrowsize=15, edge_color='darkgreen', width=1.2)
    nx.draw_networkx_labels(subgraph, sub_pos, font_size=8, font_weight='bold')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to: {os.path.abspath(output_path)}")
