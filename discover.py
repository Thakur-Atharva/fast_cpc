#!/usr/bin/env python3
"""
discover.py — Unified Causal Discovery CLI for Manufacturing Routing Data.

Supports four algorithms via flags:
  --cpc   : C-PC (Constraint-based PC from a Collection of conditioning sets)
  --kpc   : k-PC (k-Markov equivalence PC)
  --fci   : FCI  (Fast Causal Inference, allows latent confounders)
  --ges   : GES  (Greedy Equivalence Search, score-based)

Usage examples:
  python discover.py --cpc --data fake_data.csv --alpha 1e-10 --max-hubs 30
  python discover.py --kpc --data fake_data.csv --k 1 --alpha 1e-10
  python discover.py --fci --data fake_data.csv --depth 1 --alpha 0.05
  python discover.py --ges --data fake_data.csv --score-func local_score_BDeu
"""

import argparse
import os
import sys

import pandas as pd
import networkx as nx

from runners.base import (
    build_timeline_map,
    apply_temporal_filter,
    apply_transitive_reduction,
    print_results,
    save_visualization,
    save_run_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Causal Discovery for Manufacturing Routing Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Algorithm selection (mutually exclusive) ---
    algo_group = parser.add_mutually_exclusive_group(required=True)
    algo_group.add_argument("--cpc", action="store_true", help="Run C-PC algorithm")
    algo_group.add_argument("--kpc", action="store_true", help="Run k-PC algorithm")
    algo_group.add_argument("--fci", action="store_true", help="Run FCI algorithm")
    algo_group.add_argument("--ges", action="store_true", help="Run GES algorithm")

    # --- Shared arguments ---
    parser.add_argument("--data", type=str, default="fake_data.csv",
                        help="Path to the CSV data file (default: fake_data.csv)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Significance level for CI tests (default: algorithm-specific)")
    parser.add_argument("--tester", type=str, default="chisq",
                        choices=["chisq", "fisherz", "gsq"],
                        help="Independence test method (default: chisq)")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Disable chronological temporal filtering")
    parser.add_argument("--no-transitive-reduction", action="store_true",
                        help="Disable transitive reduction post-processing")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG filename (default: <algorithm>_manufacturing_dag.png)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs for CPC, k-PC, and FCI (-1 = all cores)")
    parser.add_argument("--no-save-run", action="store_true",
                        help="Disable saving run results (metadata and edges) to /runs directory")

    # --- CPC-specific ---
    parser.add_argument("--max-hubs", type=int, default=30,
                        help="[CPC] Maximum number of conditioning hubs (default: 30)")

    # --- k-PC-specific ---
    parser.add_argument("--k", type=int, default=1,
                        help="[k-PC] Maximum conditioning set size (default: 1)")
    parser.add_argument("--fast-adj", action="store_true", default=True,
                        help="[k-PC] Use fast adjacency search (default: True)")
    parser.add_argument("--no-fast-adj", action="store_true",
                        help="[k-PC] Disable fast adjacency search")

    # --- FCI-specific ---
    parser.add_argument("--depth", type=int, default=1,
                        help="[FCI] Maximum conditioning depth (default: 1)")

    # --- GES-specific ---
    parser.add_argument("--score-func", type=str, default="local_score_BDeu",
                        help="[GES] Scoring function (default: local_score_BDeu)")
    parser.add_argument("--maxP", type=int, default=None,
                        help="[GES] Maximum parents per node")
    parser.add_argument("--lambda-value", type=float, default=None,
                        help="[GES] BIC penalty hyperparameter")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load Data ---
    if not os.path.exists(args.data):
        print(f"Error: Could not find '{args.data}'", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data)
    stage_cols = [col for col in df.columns if col not in ['lotID', 'lot_id']]
    timeline_map = build_timeline_map(stage_cols)

    print(f"Loaded: {df.shape[0]} units x {len(stage_cols)} process stages from '{args.data}'")

    # --- Dispatch to runner ---
    params = {}
    if args.cpc:
        from runners.run_cpc import run_cpc
        alpha = args.alpha if args.alpha is not None else 1e-10
        params = {
            "alpha": alpha,
            "tester": args.tester,
            "max_hubs": args.max_hubs,
            "n_jobs": args.n_jobs,
        }
        result = run_cpc(
            df, stage_cols,
            alpha=alpha,
            tester=args.tester,
            max_hubs=args.max_hubs,
            n_jobs=args.n_jobs,
        )

    elif args.kpc:
        from runners.run_kpc import run_kpc
        alpha = args.alpha if args.alpha is not None else 0.05
        fast_adj = args.fast_adj and not args.no_fast_adj
        params = {
            "alpha": alpha,
            "tester": args.tester,
            "k": args.k,
            "fast_adj_search": fast_adj,
            "n_jobs": args.n_jobs,
        }
        result = run_kpc(
            df, stage_cols,
            alpha=alpha,
            tester=args.tester,
            k=args.k,
            fast_adj_search=fast_adj,
            n_jobs=args.n_jobs,
        )

    elif args.fci:
        from runners.run_fci import run_fci
        alpha = args.alpha if args.alpha is not None else 0.05
        params = {
            "alpha": alpha,
            "tester": args.tester,
            "depth": args.depth,
            "n_jobs": args.n_jobs,
        }
        result = run_fci(
            df, stage_cols,
            alpha=alpha,
            tester=args.tester,
            depth=args.depth,
            n_jobs=args.n_jobs,
        )

    elif args.ges:
        from runners.run_ges import run_ges
        params = {
            "score_func": args.score_func,
            "maxP": args.maxP,
            "lambda": args.lambda_value,
        }
        result = run_ges(
            df, stage_cols,
            score_func=args.score_func,
            maxP=args.maxP,
            lambda_value=args.lambda_value,
        )

    # --- Post-processing: build NetworkX graph for temporal filter + transitive reduction ---
    G = nx.DiGraph()
    G.add_nodes_from(stage_cols)

    # Add only directed-type edges to the DAG for post-processing
    for src, tgt, meta in result.edges:
        edge_type = meta.get("edge_type", "")
        if "directed" in edge_type or "possibly_causal" in edge_type:
            G.add_edge(src, tgt)

    raw_edge_count = G.number_of_edges()

    # Temporal filter
    if not args.no_temporal:
        G = apply_temporal_filter(G, timeline_map)
        filtered_count = raw_edge_count - G.number_of_edges()
        if filtered_count > 0:
            print(f"\n  Temporal filter removed {filtered_count} reverse-chronological edges")

    print(f"  Edges after temporal filter: {G.number_of_edges()}")

    # Transitive reduction
    if not args.no_transitive_reduction:
        pre_count = G.number_of_edges()
        G = apply_transitive_reduction(G)
        reduced_count = pre_count - G.number_of_edges()
        if reduced_count > 0:
            print(f"  Transitive reduction removed {reduced_count} redundant edges")

    # Rebuild result edges from the post-processed DAG
    # Keep non-directed edges (bidirected, undirected, etc.) as-is
    final_edges = []
    for src, tgt in G.edges():
        # Find original metadata if available
        original_meta = {"edge_type": "directed"}
        for o_src, o_tgt, o_meta in result.edges:
            if o_src == src and o_tgt == tgt:
                original_meta = o_meta
                break
        final_edges.append((src, tgt, original_meta))

    # Also include non-directed edges that weren't post-processed
    for src, tgt, meta in result.edges:
        edge_type = meta.get("edge_type", "")
        if "directed" not in edge_type and "possibly_causal" not in edge_type:
            final_edges.append((src, tgt, meta))

    result.edges = final_edges

    # --- Print results ---
    print_results(result)

    # --- Save Run Folder ---
    run_folder_path = None
    if not args.no_save_run:
        run_folder_path = save_run_results(result, args.data, params)
        print(f"  Run results saved to: {os.path.abspath(run_folder_path)}")

    # --- Visualization ---
    if run_folder_path is not None:
        run_vis_path = os.path.join(run_folder_path, "visualization.png")
        save_visualization(G, stage_cols, timeline_map, result, output_path=run_vis_path)

    # If output path is explicitly requested, save a copy there
    if args.output is not None:
        save_visualization(G, stage_cols, timeline_map, result, output_path=args.output)
    elif run_folder_path is None:
        # Fallback to local file if run saving is disabled and no output path specified
        save_visualization(G, stage_cols, timeline_map, result, output_path=None)


if __name__ == "__main__":
    main()
