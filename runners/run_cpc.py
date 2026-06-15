"""
CPC (Constraint-based PC from a Collection of conditioning sets) runner.
"""

import time
import numpy as np

from algorithms.core.cpc import CPC
from runners.base import (
    DiscoveryResult,
    auto_select_trusted_hubs,
)


def run_cpc(
    df,
    stage_cols,
    alpha=1e-10,
    tester="chisq",
    max_hubs=30,
    n_jobs=-1,
    **kwargs,
) -> DiscoveryResult:
    """
    Run the C-PC algorithm on a binary manufacturing DataFrame.

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
    max_hubs : int
        Maximum number of hubs to select for the conditioning collection.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).

    Returns
    -------
    DiscoveryResult
    """
    print(f"\n--- C-PC Configuration ---")
    print(f"  alpha={alpha}, tester={tester}, max_hubs={max_hubs}, n_jobs={n_jobs}")

    # Hub selection
    trusted_hubs = auto_select_trusted_hubs(df, stage_cols, max_choice=max_hubs)

    # Build conditioning collection: empty set + singleton hubs
    C = [set()]
    for hub in trusted_hubs:
        C.append({hub})

    # Run C-PC
    print(f"\n  Running C-PC structural learning...")
    start_time = time.time()
    learned_output, _ = CPC(
        df[stage_cols],
        tester=tester,
        I=C,
        alpha=alpha,
        data_names=stage_cols,
        n_jobs=n_jobs,
    )
    elapsed = time.time() - start_time
    print(f"  C-PC completed in {elapsed:.2f}s")

    # Extract edges from adjacency matrix
    cpc_matrix = learned_output.graph
    edges = []
    for i, src in enumerate(stage_cols):
        for j, dest in enumerate(stage_cols):
            if cpc_matrix[i, j] != 0:
                edges.append((src, dest, {"edge_type": "directed"}))

    params_str = f"alpha={alpha}, hubs={len(trusted_hubs)}, tester={tester}"

    return DiscoveryResult(
        algorithm_name="CPC",
        edges=edges,
        elapsed_seconds=elapsed,
        adj_matrix=cpc_matrix,
        params_summary=params_str,
    )
