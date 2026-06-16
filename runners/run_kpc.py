"""
k-PC (k-Markov equivalence PC) runner.
"""

import time
import numpy as np

from algorithms.core.kpc import kPC
from runners.base import DiscoveryResult


def run_kpc(
    df,
    stage_cols,
    alpha=0.05,
    tester="chisq",
    k=1,
    fast_adj_search=True,
    n_jobs=-1,
    background_knowledge=None,
    **kwargs,
) -> DiscoveryResult:
    """
    Run the k-PC algorithm on a binary manufacturing DataFrame.

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
    k : int
        Maximum conditioning set size for k-Markov equivalence.
    fast_adj_search : bool
        Use fast adjacency search (recommended for large datasets).

    Returns
    -------
    DiscoveryResult
    """
    print(f"\n--- k-PC Configuration ---")
    print(f"  k={k}, alpha={alpha}, tester={tester}, fastAdjSearch={fast_adj_search}")

    data = df[stage_cols].to_numpy()
    n = data.shape[1]

    if background_knowledge is not None:
        print(f"  Background Knowledge: active (enforcing temporal tiers)")
    print(f"\n  Running k-PC structural learning...")
    start_time = time.time()
    D, new_adj = kPC(
        data,
        tester=tester,
        k=k,
        n=n,
        alpha=alpha,
        fastAdjSearch=fast_adj_search,
        node_names=stage_cols,
        n_jobs=n_jobs,
        background_knowledge=background_knowledge,
    )
    elapsed = time.time() - start_time
    print(f"  k-PC completed in {elapsed:.2f}s")

    # Decode adjacency matrix to edges
    # Convention: graph[j,i]=1 and graph[i,j]=-1 → i --> j
    #             graph[i,j]=graph[j,i]=1 → i <-> j (bidirected)
    #             graph[j,i]=1 and graph[i,j]=2 → i o-> j
    #             graph[i,j]=graph[j,i]=-1 → i --- j (undirected)
    adj = new_adj
    edges = []
    seen = set()
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
                continue  # no edge

            seen.add(pair)
            src_name = stage_cols[i]
            tgt_name = stage_cols[j]

            if mark_ji == 1 and mark_ij == -1:
                # i --> j
                edges.append((src_name, tgt_name, {"edge_type": "directed"}))
            elif mark_ij == 1 and mark_ji == -1:
                # j --> i
                edges.append((tgt_name, src_name, {"edge_type": "directed"}))
            elif mark_ij == 1 and mark_ji == 1:
                # i <-> j (bidirected / latent confounder)
                edges.append((src_name, tgt_name, {"edge_type": "bidirected (<->)"}))
            elif mark_ij == -1 and mark_ji == -1:
                # i --- j (undirected)
                edges.append((src_name, tgt_name, {"edge_type": "undirected (---)"}))
            elif mark_ji == 1 and mark_ij == 2:
                # i o-> j
                edges.append((src_name, tgt_name, {"edge_type": "possibly_causal (o->)"}))
            elif mark_ij == 1 and mark_ji == 2:
                # j o-> i
                edges.append((tgt_name, src_name, {"edge_type": "possibly_causal (o->)"}))
            elif mark_ij == 2 and mark_ji == 2:
                # i o-o j
                edges.append((src_name, tgt_name, {"edge_type": "uncertain (o-o)"}))
            else:
                edges.append((src_name, tgt_name, {"edge_type": f"unknown ({mark_ij},{mark_ji})"}))

    params_str = f"k={k}, alpha={alpha}, tester={tester}, fastAdj={fast_adj_search}"

    return DiscoveryResult(
        algorithm_name="k-PC",
        edges=edges,
        elapsed_seconds=elapsed,
        adj_matrix=adj,
        params_summary=params_str,
    )
