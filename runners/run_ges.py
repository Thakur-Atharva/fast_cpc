"""
GES (Greedy Equivalence Search) runner.
"""

import time
import numpy as np

from causallearn.search.ScoreBased.GES import ges
from runners.base import DiscoveryResult


def run_ges(
    df,
    stage_cols,
    score_func="local_score_BDeu",
    maxP=None,
    lambda_value=None,
    **kwargs,
) -> DiscoveryResult:
    """
    Run the GES (Greedy Equivalence Search) algorithm on a binary manufacturing DataFrame.

    GES is a score-based method that greedily optimizes a scoring function
    to learn a CPDAG (completed partially directed acyclic graph).

    Parameters
    ----------
    df : pd.DataFrame
        Data with stage columns (binary 0/1).
    stage_cols : list[str]
        Column names representing process stages.
    score_func : str
        Scoring function for GES. Options:
        - 'local_score_BDeu' (recommended for discrete/binary data)
        - 'local_score_BIC'
        - 'local_score_CV_general'
    maxP : int or None
        Maximum number of parents allowed per node.
    lambda_value : float or None
        Penalty hyperparameter for BIC-based scores.

    Returns
    -------
    DiscoveryResult
    """
    print(f"\n--- GES Configuration ---")
    print(f"  score_func={score_func}, maxP={maxP}, lambda={lambda_value}")

    data = df[stage_cols].to_numpy()
    n = data.shape[1]

    print(f"\n  Running GES structural learning...")
    start_time = time.time()

    ges_kwargs = dict(
        X=data,
        score_func=score_func,
        node_names=stage_cols,
    )
    if maxP is not None:
        ges_kwargs["maxP"] = maxP
    if lambda_value is not None:
        ges_kwargs["lambda_value"] = lambda_value

    record = ges(**ges_kwargs)
    elapsed = time.time() - start_time
    print(f"  GES completed in {elapsed:.2f}s")

    # Extract results
    G_learned = record['G']
    total_score = record.get('score', None)

    # Decode adjacency matrix
    # GES convention: graph[j,i]=1 and graph[i,j]=-1 → i --> j
    #                 graph[i,j]=graph[j,i]=-1 → i --- j (undirected in CPDAG)
    adj = G_learned.graph
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
                continue

            seen.add(pair)
            src_name = stage_cols[i]
            tgt_name = stage_cols[j]

            if mark_ji == 1 and mark_ij == -1:
                # i --> j
                edges.append((src_name, tgt_name, {
                    "edge_type": "directed",
                    "score": total_score,
                }))
            elif mark_ij == 1 and mark_ji == -1:
                # j --> i
                edges.append((tgt_name, src_name, {
                    "edge_type": "directed",
                    "score": total_score,
                }))
            elif mark_ij == -1 and mark_ji == -1:
                # i --- j (undirected in CPDAG)
                edges.append((src_name, tgt_name, {
                    "edge_type": "undirected (---)",
                    "score": total_score,
                }))
            else:
                edges.append((src_name, tgt_name, {
                    "edge_type": f"unknown ({mark_ij},{mark_ji})",
                }))

    params_parts = [f"score={score_func}"]
    if maxP is not None:
        params_parts.append(f"maxP={maxP}")
    if lambda_value is not None:
        params_parts.append(f"lambda={lambda_value}")
    params_str = ", ".join(params_parts)

    return DiscoveryResult(
        algorithm_name="GES",
        edges=edges,
        elapsed_seconds=elapsed,
        adj_matrix=adj,
        graph_score=total_score,
        params_summary=params_str,
    )
