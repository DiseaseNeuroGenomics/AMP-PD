#!/usr/bin/env python
"""
Script: run_cci_liana_example.py
Author: Tereza Clarence
Date:   2025-05-13

Description:
------------
This script performs cell-cell interaction (CCI) analysis using LIANA (Python version)
on a processed single-cell AnnData object. LIANA integrates multiple ligand-receptor
methods and consensus scoring.

Input:
- Annotated AnnData object with normalized expression
- Cell type annotations in `.obs['cell_type']`

Output:
- Ranked ligand-receptor interaction table
- Optional visualizations (dotplot, heatmap)
"""

# --- Import libraries ---
import scanpy as sc
import liana as li
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load data ---
adata = sc.read_h5ad("./AMP_PD.h5ad")  # replace with your file

# Make sure cell type labels are in 'cell_type' column
assert 'cell_type' in adata.obs.columns

# --- Run LIANA pipeline ---
li.liana_wrap(
    adata,
    groupby="cell_type",
    resource="omnipath",
    expr_prop=0.1,
    min_cells=5,
    n_perms=100,
    verbose=True
)

# --- Extract results and save ---
results = adata.uns['liana_res']
results.to_csv("./liana_top_interactions.csv", index=False)

# --- Optional: Plot Top Interactions ---
top = results.sort_values("magnitude_rank", ascending=True).head(50)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=top,
    x="source",
    y="target",
    size="magnitude_rank",
    hue="magnitude_rank",
    palette="viridis",
    legend=False
)
plt.title("Top Cell-Cell Interactions (LIANA)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("./liana_top50_interactions.png")
