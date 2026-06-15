"""
FCI (Fast Causal Inference) runner.
"""

import time
import numpy as np

from algorithms.core.fci import fci_k
from runners.base import DiscoveryResult


def run_fci(
    df,
    stage_cols,
    alpha=0.05,
    tester="chisq",
    depth=1,
    n_jobs=-1,
    **kwargs,
) -> DiscoveryResult:
    """
    Run the FCI algorithm on a binary manufacturing DataFrame.

    FCI allows for latent confounders and outputs a PAG (Partial Ancestral Graph)
    with edge marks: ->, <->, o->, o-o.

    Parameters
    ----------
    df : pd.DataFrame
        Data with stage columns (binary 0/1).
    stage_cols : list[str]
        Column names representing process stages.
    alpha : float
        Significance threshold for CI tests.
    tester : str
        Independence test method ('chisq', 'fisherz', 'gsq').
    depth : int
        Maximum conditioning set depth (-1 for unlimited).

    Returns
    -------
    DiscoveryResult
    """
    print(f"\n--- FCI Configuration ---")
    print(f"  depth={depth}, alpha={alpha}, tester={tester}")

    data = df[stage_cols].to_numpy()
    n = data.shape[1]

    print(f"\n  Running FCI structural learning...")
    start_time = time.time()
    graph, color_edges = fci_k(
        data,
        independence_test_method=tester,
        alpha=alpha,
        depth=depth,
        n_jobs=n_jobs,
    )
    elapsed = time.time() - start_time
    print(f"  FCI completed in {elapsed:.2f}s")

    # Decode adjacency matrix
    # FCI uses the same convention as k-PC for the adjacency matrix
    adj = graph.graph
    edges = []
    seen = set()

    # Build node name mapping: FCI uses X1, X2, ... internally
    node_names_map = {f"X{i+1}": stage_cols[i] for i in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue

            mark_ij = int(adj[i, j])
            mark_ji = int(adj[j, i])

            if mark_ij == 0 and mark_ji == 0:
                continue

            seen.add(pair)
            src_name = stage_cols[i]
            tgt_name = stage_cols[j]

            if mark_ji == 1 and mark_ij == -1:
                edges.append((src_name, tgt_name, {"edge_type": "directed (-->)"}))
            elif mark_ij == 1 and mark_ji == -1:
                edges.append((tgt_name, src_name, {"edge_type": "directed (-->)"}))
            elif mark_ij == 1 and mark_ji == 1:
                edges.append((src_name, tgt_name, {"edge_type": "bidirected (<->)"}))
            elif mark_ij == -1 and mark_ji == -1:
                edges.append((src_name, tgt_name, {"edge_type": "undirected (---)"}))
            elif mark_ji == 1 and mark_ij == 2:
                edges.append((src_name, tgt_name, {"edge_type": "possibly_causal (o->)"}))
            elif mark_ij == 1 and mark_ji == 2:
                edges.append((tgt_name, src_name, {"edge_type": "possibly_causal (o->)"}))
            elif mark_ij == 2 and mark_ji == 2:
                edges.append((src_name, tgt_name, {"edge_type": "uncertain (o-o)"}))
            else:
                edges.append((src_name, tgt_name, {"edge_type": f"unknown ({mark_ij},{mark_ji})"}))

    params_str = f"depth={depth}, alpha={alpha}, tester={tester}"

    return DiscoveryResult(
        algorithm_name="FCI",
        edges=edges,
        elapsed_seconds=elapsed,
        adj_matrix=adj,
        params_summary=params_str,
    )
