#!/usr/bin/env python
"""
compute_tps_v1.py

Compute Transcriptomic Pathology Scores (TPS) from a single-nucleus RNA-seq
AnnData object and cell-type-specific DE results.

TPS is defined as, for each cell type and donor, the Pearson correlation between:
  - the donor's mean expression profile (per gene) after subtracting the
    cell-type-wide mean ("baseline" expression), and
  - the PD vs control logFC from differential expression (dreamlet).

Outputs:
  1) TPS_matrix.csv  (rows = cell types, columns = donors)
  2) TPS_donor.csv   (donor-level TPS: mean across cell types)
  3) Optional heatmap: TPS_heatmap.png

Author: Tereza Clarence (+ ChatGPT helper)

Example:
python compute_tps.py \
  --adata freeze2_0/your_adata_file.h5ad \
  --de_csv freeze2_0/dreamlet_all_subclass_freeze2_7_Jan2025/class_analysis/topTable_dxPD_freeze2_7_CLASS_2025_02_13_braaklb_50pFix2.csv \
  --outdir results/TPS \
  --cell_types Astro EN Endo IN Mural Myeloid OPC Oligo \
  --min_cells_per_donor 3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# TPS computation
# ---------------------------------------------------------------------------

def compute_tps_matrix(
    adata,
    df_de,
    celltype_col="derived_class2_Dec2024",
    donor_col="participant_id",
    de_celltype_col="cell_type",
    de_gene_col="geneID",
    de_logfc_col="logFC",
    cell_types=None,
    min_cells_per_donor=0,
):
    """
    Compute TPS for each (cell type, donor) pair.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-nucleus RNA-seq data with obs containing celltype_col and donor_col.
    df_de : pandas.DataFrame
        DE results with columns [de_celltype_col, de_gene_col, de_logfc_col].
    celltype_col : str
        Column in adata.obs with cell type labels.
    donor_col : str
        Column in adata.obs with donor / participant ID.
    de_celltype_col : str
        Column in df_de with cell type labels (matching celltype_col).
    de_gene_col : str
        Column with gene IDs (matching adata.var_names).
    de_logfc_col : str
        Column with log fold change values.
    cell_types : list or None
        List of cell types to process. If None, uses intersection of
        cell types present in adata and df_de.
    min_cells_per_donor : int
        If > 0, require at least this many nuclei per (cell type, donor)
        to include that cell type in TPS computation.

    Returns
    -------
    TPS_matrix : pandas.DataFrame
        DataFrame with index = cell types, columns = donors, values = TPS.
    """
    # Determine cell types to use
    adata_celltypes = adata.obs[celltype_col].unique()
    de_celltypes = df_de[de_celltype_col].unique()

    if cell_types is None:
        cell_types = sorted(set(adata_celltypes).intersection(de_celltypes))

    donors = adata.obs[donor_col].unique()
    TPS_matrix = pd.DataFrame(index=cell_types, columns=donors, dtype=float)

    # Optional: filter cell types by minimum cell count per donor
    if min_cells_per_donor > 0:
        counts = (
            adata.obs.groupby([celltype_col, donor_col])
            .size()
            .reset_index(name="n_cells")
        )
        counts_pivot = (
            counts.pivot(index=celltype_col, columns=donor_col, values="n_cells")
            .fillna(0)
        )
        keep_celltypes = counts_pivot[
            (counts_pivot >= min_cells_per_donor).all(axis=1)
        ].index
        cell_types = [ct for ct in cell_types if ct in keep_celltypes]

    print(f"Computing TPS for cell types: {cell_types}")
    print(f"Number of donors: {len(donors)}")

    # Iterate over cell types
    for cell_type in cell_types:
        print(f"  -> {cell_type}")

        # Subset AnnData to this cell type
        mask = adata.obs[celltype_col] == cell_type
        adata_sub = adata[mask]

        if adata_sub.n_obs == 0:
            print(f"    Skipping {cell_type}: no cells.")
            continue

        # Compute mean expression per gene per donor
        # to_df() returns a (cells x genes) DataFrame with var_names as columns
        expr_df = adata_sub.to_df()
        donor_series = adata_sub.obs[donor_col]
        mean_expr = expr_df.groupby(donor_series).mean()  # donors x genes

        # Subset DE to this cell type and index by gene ID
        de_sub = df_de[df_de[de_celltype_col] == cell_type].set_index(de_gene_col)

        # Restrict to genes present in both expression and DE
        common_genes = mean_expr.columns.intersection(de_sub.index)
        if len(common_genes) < 10:
            print(
                f"    Warning: {cell_type} has only {len(common_genes)} common genes, skipping."
            )
            continue

        mean_expr = mean_expr[common_genes]
        de_logfc = de_sub.loc[common_genes, de_logfc_col]

        # Baseline expression (mean across donors)
        baseline_expr = mean_expr.mean(axis=0)

        # Compute TPS per donor
        for donor in mean_expr.index:
            expr_profile = mean_expr.loc[donor]

            # Residualize by baseline expression and correlate with logFC
            try:
                r, _ = pearsonr(expr_profile - baseline_expr, de_logfc)
            except Exception:
                r = np.nan

            TPS_matrix.loc[cell_type, donor] = r

    return TPS_matrix


def compute_donor_tps(TPS_matrix, cell_types=None):
    """
    Compute donor-level TPS as mean across selected cell types.

    Parameters
    ----------
    TPS_matrix : pandas.DataFrame
        (cell types x donors) matrix of TPS values.
    cell_types : list or None
        Cell types to average over. If None, use all rows.

    Returns
    -------
    donor_TPS : pandas.Series
        Index = donors, values = mean TPS.
    """
    if cell_types is not None:
        missing = [ct for ct in cell_types if ct not in TPS_matrix.index]
        if missing:
            print(f"Warning: requested cell types not in TPS_matrix: {missing}")
        TPS_sub = TPS_matrix.loc[[ct for ct in cell_types if ct in TPS_matrix.index]]
    else:
        TPS_sub = TPS_matrix

    donor_TPS = TPS_sub.mean(axis=0)
    return donor_TPS


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tps_heatmap(TPS_matrix, out_path):
    """
    Plot a simple TPS heatmap (cell types x donors).

    Parameters
    ----------
    TPS_matrix : pandas.DataFrame
    out_path : pathlib.Path
        Output file path for PNG.
    """
    # Order donors and cell types by average TPS
    ordered_donors = TPS_matrix.mean(axis=0).sort_values().index
    ordered_celltypes = TPS_matrix.mean(axis=1).sort_values().index

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        TPS_matrix.loc[ordered_celltypes, ordered_donors],
        cmap="coolwarm",
        center=0,
        xticklabels=True,
        yticklabels=True,
    )
    plt.xlabel("Donor (participant_id)")
    plt.ylabel("Cell type")
    plt.title("Transcriptomic Pathology Score (TPS)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved TPS heatmap to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Transcriptomic Pathology Scores (TPS) "
        "from snRNA-seq AnnData and DE results."
    )
    p.add_argument(
        "--adata",
        required=True,
        help="Path to input .h5ad file with snRNA-seq data.",
    )
    p.add_argument(
        "--de_csv",
        required=True,
        help="Path to DE results CSV file.",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory for TPS CSVs and plots.",
    )
    p.add_argument(
        "--celltype_col",
        default="derived_class2_Dec2024",
        help="Column name in adata.obs with cell type labels.",
    )
    p.add_argument(
        "--donor_col",
        default="participant_id",
        help="Column name in adata.obs with donor IDs.",
    )
    p.add_argument(
        "--de_celltype_col",
        default="cell_type",
        help="Column name in DE table with cell type labels.",
    )
    p.add_argument(
        "--de_gene_col",
        default="geneID",
        help="Column name in DE table with gene identifiers.",
    )
    p.add_argument(
        "--de_logfc_col",
        default="logFC",
        help="Column name in DE table with log fold-change values.",
    )
    p.add_argument(
        "--min_cells_per_donor",
        type=int,
        default=0,
        help="Minimum number of cells per (cell type, donor) required "
        "to include that cell type (default: 0 = no filter).",
    )
    p.add_argument(
        "--cell_types",
        nargs="+",
        default=None,
        help="Optional explicit list of cell types to include "
        "(e.g. --cell_types Astro EN Endo IN Mural Myeloid OPC Oligo).",
    )
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="If set, do not generate TPS heatmap.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AnnData from: {args.adata}")
    adata = sc.read_h5ad(args.adata)

    print(f"Loading DE results from: {args.de_csv}")
    df_de = pd.read_csv(args.de_csv)

    # Ensure cell_type + geneID columns exist
    if "cell_type" not in df_de.columns and args.de_celltype_col == "cell_type":
        # For your specific file: assay -> cell_type, ID -> geneID
        if "assay" in df_de.columns:
            df_de["cell_type"] = df_de["assay"]
        if "ID" in df_de.columns:
            df_de["geneID"] = df_de["ID"]

    TPS_matrix = compute_tps_matrix(
        adata=adata,
        df_de=df_de,
        celltype_col=args.celltype_col,
        donor_col=args.donor_col,
        de_celltype_col=args.de_celltype_col,
        de_gene_col=args.de_gene_col,
        de_logfc_col=args.de_logfc_col,
        cell_types=args.cell_types,
        min_cells_per_donor=args.min_cells_per_donor,
    )

    # Save TPS matrix
    tps_matrix_path = outdir / "TPS_matrix.csv"
    TPS_matrix.to_csv(tps_matrix_path)
    print(f"Saved TPS matrix to: {tps_matrix_path}")

    # Donor-level TPS (mean across all cell types in TPS_matrix)
    donor_TPS = compute_donor_tps(TPS_matrix)
    df_donor_TPS = donor_TPS.to_frame(name="TPS_donor")
    df_donor_TPS.index.name = "participant_id"

    tps_donor_path = outdir / "TPS_donor.csv"
    df_donor_TPS.to_csv(tps_donor_path)
    print(f"Saved donor-level TPS to: {tps_donor_path}")

    # Optional heatmap
    if not args.no_plot:
        plot_tps_heatmap(TPS_matrix, outdir / "TPS_heatmap.png")


if __name__ == "__main__":
    main()
