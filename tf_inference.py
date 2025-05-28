# ==========================================================
# Script: tf_inference.py
# Author: Tereza Clarence
# Date:   2025-05-13
#
# Description:
# ------------
# This script performs transcription factor (TF) activity inference using decoupler
# on AMP-PD subclass-level pseudobulk differential expression data. TF activity is
# inferred per cell class using the CollecTRI regulatory network. Output includes
# TF activity scores, significance values, and a heatmap of top TFs per class.
# ==========================================================

import scanpy as sc
import decoupler as dc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pypath import omnipath
import omnipath
import seaborn as sns

# Plotting options
sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
random.seed(18)

# Retrieve CollecTRI gene regulatory network
collectri = dc.get_collectri(organism='human', split_complexes=False)

# ==========================================================
# utils.py
# ==========================================================

def get_dict_types(act, obs):
    vs = np.unique(np.hstack([act.columns, obs.columns]))
    v_dict = {k: i for i, k in enumerate(vs)}
    types = (~np.isin(vs, act.columns)) * 1
    return v_dict, types

def net_to_edgelist(v_dict, net):
    edges = []
    for i in net.index:
        source, target = net.loc[i, 'source'], net.loc[i, 'target']
        edge = [v_dict[source], v_dict[target]]
        edges.append(edge)
    return edges

def get_g(act, obs, net):
    import igraph as ig
    v_dict, types = get_dict_types(act, obs)
    edges = net_to_edgelist(v_dict, net)
    g = ig.Graph(edges=edges, directed=True)
    g.es['weight'] = net['weight'].values
    g.vs['type'] = types
    g.vs['label'] = list(v_dict.keys())
    g.vs['shape'] = np.where(types, 'circle', 'square')
    return g

def min_max_neg1_to_1(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0
    return 2 * ((series - min_val) / (max_val - min_val)) - 1

# ==========================================================
# plotting.py
# ==========================================================

def plot_tf_network(collectri, mat, tf_acts, tf_list, title):
    dc.plot_network(
        net=collectri,
        obs=mat,
        act=tf_acts,
        n_sources=tf_list,
        n_targets=15,
        node_size=100,
        figsize=(7, 7),
        c_pos_w='darkgreen',
        c_neg_w='darkred',
        vcenter=True
    )
    plt.title(title)
    plt.show()

# ==========================================================
# Main Analysis
# ==========================================================

bla = pd.read_csv('./topTable_dxPD_DEresults.csv') # DEG results
bla = bla.drop(columns=["Unnamed: 0"], errors="ignore")

all_results_df = pd.DataFrame()
all_results_df['baseMean'] = bla['AveExpr']
all_results_df['log2FoldChange'] = bla['logFC']
all_results_df['lfcSE'] = bla["logFC"] / bla["t"]
all_results_df['stat'] = bla['t'].astype(float)
all_results_df['pvalue'] = bla['P.Value']
all_results_df['padj'] = bla['adj.P.Val']
all_results_df['class'] = bla['assay']
all_results_df['ID'] = bla['ID']
all_results_df = all_results_df.set_index("ID")

# Store TF results
tf_acts_df = pd.DataFrame()
tf_acts_dfp = pd.DataFrame()
classes = all_results_df['class'].unique()
ord_classes = ['Adaptive', 'Astro','Chol', 'EN', 'Endo','Ependymal', 'IN', 'IN_MSN', 'Mural', 'Myeloid','OPC', 'Oligo']

for celltype in ord_classes:
    subset = all_results_df[all_results_df['class'] == celltype]
    mat = subset[['stat']].T.rename(index={'stat': celltype})
    tf_acts, tf_pvals = dc.run_consensus(mat=mat, net=collectri)
    tf_acts_long = tf_acts.reset_index(names='class').melt(id_vars='class', var_name='TF', value_name='TF_activity_score')
    tf_acts_longp = tf_pvals.reset_index(names='class').melt(id_vars='class', var_name='TF', value_name='TF_p_value')
    tf_acts_df = pd.concat([tf_acts_df, tf_acts_long], ignore_index=False)
    tf_acts_dfp = pd.concat([tf_acts_dfp, tf_acts_longp], ignore_index=False)

# Normalize and calculate specificity
tf_acts_df['TF_activity_norm'] = tf_acts_df.groupby('class')['TF_activity_score'].transform(min_max_neg1_to_1)
tf_acts_df['TF_p_value'] = tf_acts_dfp['TF_p_value']

min_val = tf_acts_df['TF_activity_score'].min()
shift = abs(min_val) + 1e-6 if min_val <= 0 else 0
tf_acts_df['shifted_activity'] = tf_acts_df['TF_activity_score'] + shift
tf_acts_df['TF_specificity'] = tf_acts_df.groupby('TF')['shifted_activity'].transform(lambda x: x / x.sum())

# Heatmap of top TFs
ORD = ord_classes
NUM_TopTFs = 5
top_tfs = {}
for sc in ORD:
    sub_df = tf_acts_df[tf_acts_df["class"] == sc]
    if sub_df.empty: continue
    top_df = sub_df.sort_values(by=["TF_activity_norm", "TF_specificity"], ascending=[False, False]).head(NUM_TopTFs)
    top_tfs[sc] = top_df["TF"].tolist()

tf_order = []
seen = set()
for tfs in top_tfs.values():
    for tf in tfs:
        if tf not in seen:
            tf_order.append(tf)
            seen.add(tf)

subset_df = tf_acts_df[tf_acts_df["TF"].isin(tf_order)]
heatmap_df = subset_df.pivot(index="class", columns="TF", values="TF_activity_norm").reindex(index=ORD).reindex(columns=tf_order)

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_df, cmap="coolwarm", annot=False, linewidths=0.8, linecolor="grey", xticklabels=True, yticklabels=True)
plt.xticks(rotation=90)
plt.title("Top TFs Per Class — Primary: TF_activity_norm, Secondary: TF_specificity")
plt.xlabel("TF")
plt.ylabel("Class")
plt.tight_layout()
plt.show()
