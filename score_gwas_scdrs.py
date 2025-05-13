#!/usr/bin/env python3
"""
Script: score_gwas_scdrs.py
Author: Tereza Clarence
Date:   2025-05-13

Description:
------------
This script loads a single-cell AnnData object and applies the SCDRS framework
to score each cell for a Parkinson's disease GWAS gene set (e.g., age of onset).
It performs downstream group-wise analyses by taxonomy level and exports results.

Usage:
------
$ python score_gwas_scdrs.py

Dependencies:
-------------
- scanpy
- pegasus
- pandas
- numpy
- matplotlib
- seaborn
- scdrs
- pynndescent
"""

# --- Import Libraries ---

# Core analysis
import scanpy as sc
import pegasus as pg
import anndata as ad
import scdrs

# Plotting (not used here but loaded if needed)
import matplotlib.pyplot as plt
import seaborn as sns

# System & utility
import os
import sys
import csv
import re
import glob
import numpy as np
import pandas as pd
from scipy.stats import zscore


# --- Set Parameters ---

# Input path
PATH = "./"
FILENAME = "AMP_PD.h5ad"
OUTPUT_PREFIX = "AMPPD"

# --- Load Data ---

adata = sc.read_h5ad(os.path.join(PATH, FILENAME))
print(adata)

# --- Preprocess Data for SCDRS ---

scdrs.preprocess(adata, n_mean_bin=20, n_var_bin=20, copy=False)

# --- Load Gene Set ---

dict_gs = scdrs.util.load_gs(
    "./gwas_geneset.gs",
    src_species="human",
    dst_species="human",
    to_intersect=adata.var_names,
)

# Select only Parkinson's disease  gene set
dict_you_want = {"pd": dict_gs["pd"]}

# --- Score Cells ---

dict_df_score = {}

for trait, (gene_list, gene_weights) in dict_you_want.items():
    print(f"Scoring for trait: {trait}")
    dict_df_score[trait] = scdrs.score_cell(
        data=adata,
        gene_list=gene_list,
        gene_weight=gene_weights,
        ctrl_match_key="mean_var",
        n_ctrl=300,
        weight_opt="vs",
        return_ctrl_raw_score=False,
        return_ctrl_norm_score=True,
        verbose=False,
    )

# --- Define Grouping Variables and Export Results ---

group_levels = [
    "class",
    "subclass",
    "subtype"]

for trait, df_score in dict_df_score.items():
    for group in group_levels:
        print(f"Analyzing trait '{trait}' at group level: {group}")
        df_stats = scdrs.method.downstream_group_analysis(
            adata=adata,
            df_full_score=df_score,
            group_cols=[group],
        )[group]

        output_base = f"{OUTPUT_PREFIX}_{group}_{trait}"
        df_score.to_csv(f"{output_base}.csv", sep="\t", quoting=csv.QUOTE_NONE, escapechar=" ")
        df_stats.to_csv(f"{output_base}_summary.csv", sep="\t", quoting=csv.QUOTE_NONE, escapechar=" ")

print("✔️ Analysis complete. All scores and summaries saved.")
