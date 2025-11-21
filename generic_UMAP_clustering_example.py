#!/usr/bin/env python
"""
Global clustering and UMAP embedding for large snRNA-seq datasets.

This script demonstrates how to:
  1) Subset an AnnData .h5ad file on disk to protein-coding autosomal genes,
  2) Select highly variable genes (HVGs),
  3) Run PCA + Harmony batch correction,
  4) Build a kNN graph and compute UMAP,
  5) Run Leiden clustering at two resolutions (0.10 and 0.20),
  6) Save the resulting AnnData object.

Dependencies:
  - pegasus            (https://github.com/lilab-bcb/pegasus)
  - scanpy             (https://scanpy.readthedocs.io)
  - pge (Pegasus extras; optional, or replace with your own utilities)

This script assumes:
  - Input is a large .h5ad file with raw counts in X,
  - .var contains: gene_type, gene_chrom (chromosome annotation),
  - .obs contains: a batch key (e.g. "Brain_bank" or "Source").

Author: Tereza Clarence, Donghoon Lee

### Global clustering & UMAP

Example usage:

```bash
python scripts/global_clustering_umap.py \
  --orig-h5ad data/AMP_PD_freeze2_all.h5ad \
  --subset-h5ad data/AMP_PD_freeze2_autosome_pc.h5ad \
  --output-h5ad results/AMP_PD_freeze2_global_umap.h5ad \
  --batch-key Brain_bank
"""

import argparse
from pathlib import Path

import pegasus as pg
import scanpy as sc
import h5py

import numpy as np

# If pge is part of your repo, import it; otherwise replace with your own utilities
import pge  # expected to provide read_everything_but_X() and ondisk_subset(), scanpy_hvf_h5ad()


def subset_on_disk(
    orig_h5ad: str,
    new_h5ad: str,
    chunk_size: int = 500_000,
):
    """
    Create an on-disk subset of an AnnData file, keeping only protein-coding autosomal genes.
    """
    print(f"[INFO] Reading metadata from {orig_h5ad}")
    adata = pge.read_everything_but_X(orig_h5ad)

    # Keep all cells
    subset_obs = (adata.obs_names != None).tolist()

    # Keep autosomal protein-coding genes (exclude MT, X, Y)
    subset_var = (
        (adata.var["gene_type"] == "protein_coding")
        & (~adata.var["gene_chrom"].isin(["MT", "X", "Y"]))
    ).tolist()

    print("[INFO] Subsetting on disk to protein-coding autosomal genes...")
    pge.ondisk_subset(
        orig_h5ad=orig_h5ad,
        new_h5ad=new_h5ad,
        subset_obs=subset_obs,
        subset_var=subset_var,
        chunk_size=chunk_size,
        raw=False,
    )

    # Optional: quick structural check
    with h5py.File(new_h5ad, "r") as f:
        print("[INFO] New X dataset structure:")
        f["X"].visititems(print)

    print(f"[DONE] Wrote subsetted file to {new_h5ad}")


def run_global_clustering_umap(
    input_h5ad: str,
    output_h5ad: str,
    batch_key: str = "Brain_bank",
    n_top_genes: int = 6000,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    n_pcs: int = 30,
    k_graph: int = 100,
    n_neighbors_umap: int = 100,
):
    """
    Perform global clustering and UMAP on a (potentially large) snRNA-seq dataset.
    """

    # -------------------------------------------------------------------------
    # 1. Select HVGs using Scanpy-style HVG function
    # -------------------------------------------------------------------------
    print(f"[INFO] Selecting HVGs from {input_h5ad}")
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
    print(f"[INFO] Selected {len(hvg)} HVGs")

    # Optional: save HVG list
    hvg_txt = Path(output_h5ad).with_suffix(".hvg.txt")
    with open(hvg_txt, "w") as fp:
        for g in hvg:
            fp.write(f"{g}\n")
    print(f"[INFO] Saved HVG list to {hvg_txt}")

    # Re-read to ensure consistent ordering
    with open(hvg_txt) as f:
        hvg = [line.rstrip() for line in f]

    # -------------------------------------------------------------------------
    # 2. Load data via Pegasus and mark HVGs
    # -------------------------------------------------------------------------
    print(f"[INFO] Loading data from {input_h5ad}")
    data = pg.read_input(input_h5ad, genome="GRCh38", modality="rna")
    print(data)

    data.var["highly_variable_features"] = False
    data.var.loc[data.var.index.isin(hvg), "highly_variable_features"] = True

    print("[INFO] HVG flag summary:")
    print(data.var["highly_variable_features"].value_counts())
    print("[INFO] HVGs by chromosome:")
    print(data.var[data.var["highly_variable_features"] == True]["gene_chrom"].value_counts())

    # -------------------------------------------------------------------------
    # 3. PCA, regression, Harmony integration
    # -------------------------------------------------------------------------
    print("[INFO] Running PCA")
    pg.pca(data, n_components=n_pcs)
    pg.elbowplot(data)
    npc = min(data.uns["pca_ncomps"], n_pcs)
    print(f"[INFO] Using {npc} PCs for downstream steps")

    print("[INFO] Regressing out technical covariates and running Harmony")
    pg.regress_out(data, attrs=["n_counts", "percent_mito", "cycle_diff"])
    pg.run_harmony(
        data,
        batch=batch_key,
        rep="pca_regressed",
        max_iter_harmony=20,
        n_comps=npc,
    )

    print("[INFO] Building kNN graph")
    pg.neighbors(
        data,
        rep="pca_regressed_harmony",
        use_cache=False,
        dist="l2",
        K=k_graph,
        n_comps=npc,
    )

    # -------------------------------------------------------------------------
    # 4. UMAP and Leiden clustering
    # -------------------------------------------------------------------------
    print("[INFO] Computing UMAP")
    pg.umap(
        data,
        rep="pca_regressed_harmony",
        n_neighbors=n_neighbors_umap,
        rep_ncomps=npc,
    )

    adata = data.to_anndata()

    print("[INFO] Running Leiden clustering (res 0.10, 0.20)")
    sc.pp.neighbors(adata, use_rep="X_pca_regressed_harmony")
    sc.tl.leiden(adata, key_added="leiden_res0_10", resolution=0.10)
    sc.tl.leiden(adata, key_added="leiden_res0_20", resolution=0.20)

    # Save final AnnData
    adata.write_h5ad(output_h5ad)
    print(f"[DONE] Saved clustered AnnData to {output_h5ad}")

    # Optional: UMAP figure coloured by Leiden clusters
    sc.pl.umap(
        adata,
        color=["leiden_res0_20", "leiden_res0_10"],
        legend_loc="on data",
        frameon=False,
        legend_fontsize=5,
        legend_fontoutline=1,
        title=["Leiden res 0.20", "Leiden res 0.10"],
        size=1,
        wspace=0,
        ncols=2,
        save="_global_clustering.png",
    )
    print("[DONE] UMAP plot saved (scanpy default location)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global clustering + UMAP for large snRNA-seq AnnData files."
    )
    parser.add_argument(
        "--orig-h5ad",
        type=str,
        required=True,
        help="Path to the original (large) AnnData .h5ad file.",
    )
    parser.add_argument(
        "--subset-h5ad",
        type=str,
        required=True,
        help="Path for the on-disk subset (protein-coding autosomal genes).",
    )
    parser.add_argument(
        "--output-h5ad",
        type=str,
        required=True,
        help="Path for the final clustered .h5ad file (with UMAP + Leiden).",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default="Brain_bank",
        help="Batch key used for Harmony integration (e.g. brain bank / source).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Chunk size for on-disk subsetting.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=6000,
        help="Number of HVGs to select globally.",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=30,
        help="Number of principal components to retain.",
    )
    parser.add_argument(
        "--k-graph",
        type=int,
        default=100,
        help="Number of neighbors (K) for kNN graph construction.",
    )
    parser.add_argument(
        "--n-neighbors-umap",
        type=int,
        default=100,
        help="Number of neighbors for UMAP.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1) On-disk subset (protein-coding autosomes)
    subset_on_disk(
        orig_h5ad=args.orig_h5ad,
        new_h5ad=args.subset_h5ad,
        chunk_size=args.chunk_size,
    )

    # 2) Global clustering and UMAP on the subsetted file
    run_global_clustering_umap(
        input_h5ad=args.subset_h5ad,
        output_h5ad=args.output_h5ad,
        batch_key=args.batch_key,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        k_graph=args.k_graph,
        n_neighbors_umap=args.n_neighbors_umap,
    )
