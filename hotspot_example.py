#!/usr/bin/env python3
"""
Script: hotspot_example.py
Author: Tereza Clarence
Date:   2025-05-13

Description:
------------
This script runs Hotspot analysis on a subset of microglia/PVM cells using a preprocessed
AnnData object. It matches updated cell annotations from a metadata file, computes autocorrelations,
derives gene modules, and saves the output scores and module assignments.

Input:
------
- h5ad file with PCA/UMAP already run
- metadata .csv with updated derived_class/subclass/subtype

Output:
-------
- Filtered .h5ad file with Hotspot results
- CSV with module assignments and module scores

Requirements:
-------------
- scanpy
- pegasus
- hotspot
- pandas, numpy
"""

# --- Imports ---

# Core analysis
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
from scipy import sparse

# System
import gc
import sys
from pathlib import Path

# Pegasus extensions (custom path)
sys.path.append("pge")  # Replace with your path to custom Pegasus tools
import pge
pge.info()

# Hotspot
import hotspot
import scipy.sparse as sp
from anndata import AnnData


# --- Step 1: Load Data ---

adata = sc.read_h5ad('./Myeloid.h5ad')
print(adata)


# --- Step 2: Filter Lowly Expressed Genes ---

adata2 = sc.pp.filter_genes(adata, min_cells=10, copy=True)


# --- Step 3: Hotspot Analysis ---

hs = hotspot.Hotspot(
    adata,
    layer_key="counts",  # assumes raw counts layer
    model="danb",
    latent_obsm_key="X_pca_regressed_harmony",  # harmony-corrected PCA
    umi_counts_obs_key="total_counts"
)

# Build graph and compute autocorrelations
hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
hs_results = hs.compute_autocorrelations()
print(hs_results.head())

# Select top 1k genes 
hs_genes = hs_results.query("FDR < 0.05").sort_values("Z", ascending=False).head(1000).index
local_correlations = hs.compute_local_correlations(hs_genes, jobs=4)

# Create modules
modules = hs.create_modules(min_gene_threshold=30, core_only=True, fdr_threshold=0.05)
print(modules.head())
modules.to_csv("./Myeloid_modules_top1000.csv")

# Save module scores
module_scores = hs.calculate_module_scores()
pd.DataFrame(hs.module_scores).to_csv("./Myeloid_module_scores_top1000.csv")

# Save Hotspot object
hs.write_h5ad("./Myeloid_with_hs1000genes.h5ad")

print("✔ Hotspot analysis complete. Outputs saved.")
