#!/usr/bin/env python3
"""
Script: prepare_and_integrate_AMPPD.py
Author: Tereza Clarence
Date:   2025-05-13

Description:
------------
This script prepares and filters single-nucleus RNA-seq data (AMP PD),
performs HVG selection, dimensionality reduction (PCA), batch correction (Harmony),
and neighbor/UMAP/Leiden clustering for downstream integration and visualization.

Input:
------
- AMP_PD_before.h5ad (raw)
- Outputs cleaned file: AMPPD_prep_clean_lcg_autosome.h5ad
- Outputs PCA/UMAP-integrated file: AMPPD_prep_clean_lcg_autosome_30pcaUMAP.h5ad

Requirements:
-------------
- scanpy
- pegasus
- harmony-pytorch
- pandas, numpy, h5py, seaborn
"""

# --- Imports ---

# Core tools
import scanpy as sc
import pegasus as pg
import scanpy.external as sce
import anndata as ad

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Data utilities
import numpy as np
import pandas as pd
import h5py
from scipy import sparse, stats

# File & system
import sys
import gc
from pathlib import Path

# Batch correction
from harmony import harmonize

# Pegasus extensions
sys.path.append("pge")  # Your local path to a set of customized pegasus functions
import pge
pge.info()

# --- Input / Output Parameters ---

raw_h5ad = "./AMP_PD_before.h5ad"
clean_h5ad = "./AMPPD_prep_clean_lcg_autosome.h5ad"
hvg_output_txt = "./AMP_PD_hvg6k.txt"
final_output_h5ad = "./AMPPD_prep_clean_lcg_autosome_30pcaUMAP.h5ad"

hvg_n_genes = 6000
batch_key = "Brain_bank"
n_pcs = 30
n_neighbors = 100
chunk_size = 500000


# --- Step 1: Subset Protein-Coding Genes on Autosomes ---

adata = pge.read_everything_but_X(raw_h5ad)

subset_obs = (adata.obs_names != None).tolist()
subset_var = ((adata.var["gene_type"] == "protein_coding") &
              (~adata.var["gene_chrom"].isin(["MT", "X", "Y"]))).tolist()

pge.ondisk_subset(
    orig_h5ad=raw_h5ad,
    new_h5ad=clean_h5ad,
    subset_obs=subset_obs,
    subset_var=subset_var,
    chunk_size=chunk_size,
    raw=False
)

# Optional: Inspect structure
with h5py.File(clean_h5ad, "r") as f:
    f["X"].visititems(print)


# --- Step 2: HVG Selection with Scanpy Method ---

hvg = pge.scanpy_hvf_h5ad(
    h5ad_file=clean_h5ad,
    flavor="cell_ranger",
    batch_key=batch_key,
    n_top_genes=hvg_n_genes,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
    protein_coding=True,
    autosome=True
)
print(f"Selected {len(hvg)} highly variable genes.")

# Save HVGs to file
with open(hvg_output_txt, "w") as fp:
    for gene in hvg:
        fp.write(f"{gene}\n")
print("✔ HVG gene list saved.")

# Reload for safety
with open(hvg_output_txt) as f:
    hvg = [line.strip() for line in f]


# --- Step 3: Load Cleaned Data and Apply HVGs ---

data = pg.read_input(clean_h5ad, genome="GRCh38", modality="rna")
data.var["highly_variable_features"] = False
data.var.loc[data.var.index.isin(hvg), "highly_variable_features"] = True

print(data.var["highly_variable_features"].value_counts())
print(data.var[data.var["highly_variable_features"]].gene_chrom.value_counts())


# --- Step 4: PCA and Harmony Integration ---

pg.pca(data, n_components=n_pcs)
pg.elbowplot(data)

npc = min(data.uns["pca_ncomps"], n_pcs)
print(f"Using {npc} PCs")

pg.regress_out(data, attrs=["n_counts", "percent_mito", "cycle_diff"])
pg.run_harmony(data, batch=batch_key, rep="pca_regressed", max_iter_harmony=20, n_comps=npc)
pg.neighbors(data, rep="pca_regressed_harmony", use_cache=False, dist="l2", K=n_neighbors, n_comps=npc)
pg.umap(data, rep="pca_regressed_harmony", n_neighbors=n_neighbors, rep_ncomps=npc)


# --- Step 5: Leiden Clustering ---

adata = data.to_anndata()
sc.pp.neighbors(adata, use_rep="X_pca_regressed_harmony")
sc.tl.leiden(adata, key_added="leiden_res0_10", resolution=0.10)
sc.tl.leiden(adata, key_added="leiden_res0_20", resolution=0.20)

adata.write_h5ad(final_output_h5ad)
print("✔ Data saved with UMAP + Leiden clustering")


# --- Step 6: UMAP Visualization ---

sc.pl.umap(
    adata,
    color=["leiden_res0_20", "leiden_res0_10"],
    legend_loc="on data",
    frameon=False,
    legend_fontsize=5,
    legend_fontoutline=1,
    size=1,
    wspace=0,
    ncols=2,
    save=f"{final_output_h5ad}.png"
)
