#!/usr/bin/env python
"""
Iterative class-restricted clustering and pseudobulk aggregation.

This script demonstrates how to:
  1) Take class-level subsets of a large snRNA-seq dataset (AnnData .h5ad),
  2) Recompute HVGs, PCA, Harmony, kNN, UMAP within each class,
  3) Run Leiden clustering at two resolutions (0.10, 0.20),
  4) Save updated AnnData objects and corresponding pseudobulk profiles.

Dependencies:
  - pegasus            (https://github.com/lilab-bcb/pegasus)
  - scanpy             (https://scanpy.readthedocs.io)
  - pge (Pegasus extras; optional, or replace with your own utilities)

This example assumes that you have already:
  - Run a global clustering,
  - Assigned a class label (e.g., "leiden_res0_10_class"),
  - Saved one .h5ad file per class to disk.

Author: Tereza Clarence, Donghoon Lee

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import pegasus as pg
import scanpy as sc
from anndata import AnnData

# If pge is part of your repo, import it; otherwise replace with your own helpers
import pge  # expected to provide scanpy_hvf_h5ad() and pb_agg_by_cluster()


def run_iterative_clustering(
    input_prefix: str,
    class_key: str,
    class_list,
    batch_key: str = "Source",
    n_top_genes: int = 3000,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    n_pcs: int = 30,
    k_graph: int = 100,
    n_neighbors_umap: int = 15,
    output_dir: str = "./iterative_clustering_output",
):
    """
    Perform iterative clustering within each major class and generate pseudobulk profiles.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls in class_list:
        print(f"[INFO] Running iterative clustering for class: {cls} (key: {class_key})")

        # ---------------------------------------------------------------------
        # 1. Define input / output file names
        # ---------------------------------------------------------------------
        input_h5ad = f"{input_prefix}_{class_key}_{cls}.h5ad"
        clustered_h5ad = output_dir / f"{input_prefix}_{class_key}_{cls}_hvg{n_top_genes}_umap.h5ad"
        raw_h5ad = output_dir / f"{input_prefix}_{class_key}_{cls}_hvg{n_top_genes}_umap_raw.h5ad"

        # ---------------------------------------------------------------------
        # 2. Recompute HVGs within class
        # ---------------------------------------------------------------------
        print(f"[INFO] Recomputing HVGs for {cls} from {input_h5ad}")
        hvg = pge.scanpy_hvf_h5ad(
            h5ad_file=input_h5ad,
            flavor="cell_ranger",
            batch_key=batch_key,
            n_top_genes=n_top_genes,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            protein_coding=True,
            autosome=True,
        )
        print(f"[INFO] Selected {len(hvg)} HVGs for class {cls}")

        # ---------------------------------------------------------------------
        # 3. Load data and set HVGs
        # ---------------------------------------------------------------------
        data = pg.read_input(input_h5ad, genome="GRCh38", modality="rna")

        data.var["highly_variable_features"] = False
        data.var.loc[data.var.index.isin(hvg), "highly_variable_features"] = True

        # ---------------------------------------------------------------------
        # 4. PCA, regress out technical covariates, Harmony integration
        # ---------------------------------------------------------------------
        print(f"[INFO] Running PCA, regression, Harmony for class {cls}")
        pg.pca(data, n_components=n_pcs)

        pg.regress_out(data, attrs=["n_counts", "percent_mito", "cycle_diff"])
        pg.run_harmony(
            data,
            batch=batch_key,
            rep="pca_regressed",
            max_iter_harmony=20,
            n_comps=n_pcs,
        )

        pg.neighbors(
            data,
            rep="pca_regressed_harmony",
            use_cache=False,
            dist="l2",
            K=k_graph,
            n_comps=n_pcs,
        )

        # ---------------------------------------------------------------------
        # 5. UMAP and Leiden clustering
        # ---------------------------------------------------------------------
        print(f"[INFO] Computing UMAP and Leiden clustering for class {cls}")
        pg.umap(
            data,
            rep="pca_regressed_harmony",
            n_neighbors=n_neighbors_umap,
            rep_ncomps=n_pcs,
        )

        adata = data.to_anndata()
        sc.pp.neighbors(adata, use_rep="X_pca_regressed_harmony")
        sc.tl.leiden(adata, key_added="subcl_leiden_res0_10", resolution=0.10)
        sc.tl.leiden(adata, key_added="subcl_leiden_res0_20", resolution=0.20)

        # Save clustered AnnData
        adata.write_h5ad(clustered_h5ad)
        print(f"[INFO] Saved clustered AnnData to {clustered_h5ad}")

        # Optional: UMAP plot coloured by Leiden clusters
        sc.pl.umap(
            adata,
            color=["subcl_leiden_res0_10", "subcl_leiden_res0_20"],
            legend_loc="on data",
            frameon=False,
            legend_fontsize=5,
            legend_fontoutline=1,
            title=[f"{class_key}: {cls}"],
            size=1,
            wspace=0,
            ncols=2,
            save=f"_{cls}_subclustering.png",
        )

        # ---------------------------------------------------------------------
        # 6. Attach raw counts layer and write "raw" AnnData
        # ---------------------------------------------------------------------
        print(f"[INFO] Preparing raw layer for pseudobulk aggregation ({cls})")
        if "counts" in adata.layers:
            raw_counts = adata.layers["counts"]
        else:
            # Fallback: assume X contains raw or near-raw counts
            raw_counts = adata.X

        # Ensure dense matrix for AnnData(raw) if needed; for very large data,
        # users may prefer to keep this sparse.
        if hasattr(raw_counts, "toarray"):
            raw_counts_dense = raw_counts.toarray()
        else:
            raw_counts_dense = raw_counts

        raw_adata = AnnData(raw_counts_dense)
        adata.raw = raw_adata
        adata.write_h5ad(raw_h5ad)
        print(f"[INFO] Saved AnnData with raw counts to {raw_h5ad}")

        # ---------------------------------------------------------------------
        # 7. Pseudobulk aggregation (Leiden res 0.10)
        # ---------------------------------------------------------------------
        print(f"[INFO] Computing pseudobulk (res 0.10) for class {cls}")
        pb_output_010 = output_dir / f"pbagg_scaled_{cls}_leiden0_10.csv"
        cluster_label_010 = "subcl_leiden_res0_10"

        pb_010 = pge.pb_agg_by_cluster(
            h5ad_file=str(raw_h5ad),
            cluster_label=cluster_label_010,
            robust_var_label=None,
            log1p=True,
        )

        pb_010_scaled = (pb_010 - np.mean(pb_010, axis=0)) / np.std(pb_010, axis=0)
        pb_010_scaled.to_csv(pb_output_010)
        print(f"[INFO] Saved pseudobulk (res 0.10) to {pb_output_010}")

        # ---------------------------------------------------------------------
        # 8. Pseudobulk aggregation (Leiden res 0.20)
        # ---------------------------------------------------------------------
        print(f"[INFO] Computing pseudobulk (res 0.20) for class {cls}")
        pb_output_020 = output_dir / f"pbagg_scaled_{cls}_leiden0_20.csv"
        cluster_label_020 = "subcl_leiden_res0_20"

        pb_020 = pge.pb_agg_by_cluster(
            h5ad_file=str(raw_h5ad),
            cluster_label=cluster_label_020,
            robust_var_label=None,
            log1p=True,
        )

        pb_020_scaled = (pb_020 - np.mean(pb_020, axis=0)) / np.std(pb_020, axis=0)
        pb_020_scaled.to_csv(pb_output_020)
        print(f"[INFO] Saved pseudobulk (res 0.20) to {pb_output_020}")

        print(f"[DONE] Class {cls} completed.\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative class-restricted clustering and pseudobulk aggregation."
    )
    parser.add_argument(
        "--input-prefix",
        type=str,
        required=True,
        help=(
            "Prefix used when saving class-specific .h5ad files. "
            "Script expects files of the form: <prefix>_<class_key>_<class>.h5ad"
        ),
    )
    parser.add_argument(
        "--class-key",
        type=str,
        default="leiden_res0_10_class",
        help="Name of the class-level annotation used in file naming.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=True,
        help="List of class labels to process (e.g. Mural Micro_PVM EN IN).",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default="Source",
        help="Batch key used for Harmony integration (e.g. brain bank).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./iterative_clustering_output",
        help="Directory to store output .h5ad files and pseudobulk CSVs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_iterative_clustering(
        input_prefix=args.input_prefix,
        class_key=args.class_key,
        class_list=args.classes,
        batch_key=args.batch_key,
        output_dir=args.output_dir,
    )
