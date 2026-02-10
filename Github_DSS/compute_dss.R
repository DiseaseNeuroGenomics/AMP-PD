suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(matrixStats)
})

compute_dss_all_celltypes <- function(
  pbSCE,
  de,
  celltypes,
  dx_col = "dxNEW",
  ctrl_labels = c("CTRL", "Control"),
  de_celltype_col = "assay",
  de_gene_col = "ID",
  weight_col_prefer = c("t", "logFC"),
  baseline = c("control_mean", "all_mean"),
  zscore_genes = TRUE,
  min_common_genes = 50,
  verbose = TRUE
) {

  baseline <- match.arg(baseline)

  meta <- as.data.frame(colData(pbSCE))
  meta$donor <- rownames(meta)
  meta$is_ctrl <- meta[[dx_col]] %in% ctrl_labels

  donors <- colnames(pbSCE)
  DSS_mat <- matrix(
    NA_real_,
    nrow = length(celltypes),
    ncol = length(donors),
    dimnames = list(celltypes, donors)
  )

  for (ct in celltypes) {
    if (!ct %in% assayNames(pbSCE)) next
    if (verbose) message("Processing cell type: ", ct)

    X <- t(as.matrix(assay(pbSCE, ct)))
    rownames(X) <- colnames(assay(pbSCE, ct))
    colnames(X) <- rownames(assay(pbSCE, ct))

    meta_ct <- meta[match(rownames(X), meta$donor), , drop = FALSE]

    de_ct <- de[de[[de_celltype_col]] == ct, , drop = FALSE]
    if (nrow(de_ct) == 0) next

    wcol <- weight_col_prefer[weight_col_prefer %in% colnames(de_ct)][1]
    if (is.na(wcol)) stop("No valid weight column found.")

    w <- de_ct[[wcol]]
    names(w) <- de_ct[[de_gene_col]]
    w <- w[!is.na(w)]
    w <- w[!duplicated(names(w))]

    common <- intersect(colnames(X), names(w))
    if (length(common) < min_common_genes) next

    Xc <- X[, common, drop = FALSE]
    wc <- w[common]

    if (baseline == "control_mean") {
      ctrl_idx <- which(meta_ct$is_ctrl)
      b <- if (length(ctrl_idx) >= 2)
        colMeans(Xc[ctrl_idx, , drop = FALSE], na.rm = TRUE)
      else
        colMeans(Xc, na.rm = TRUE)
    } else {
      b <- colMeans(Xc, na.rm = TRUE)
    }

    Xc <- sweep(Xc, 2, b, "-")

    if (zscore_genes) {
      mu_g <- colMeans(Xc, na.rm = TRUE)
      sd_g <- colSds(Xc, na.rm = TRUE)
      sd_g[sd_g == 0] <- 1
      Xc <- sweep(sweep(Xc, 2, mu_g, "-"), 2, sd_g, "/")
    }

    DSS_mat[ct, rownames(Xc)] <- row_cor_pearson(Xc, wc)
  }

  list(
    DSS_matrix = as.data.frame(DSS_mat),
    DSS_donor = colMeans(DSS_mat, na.rm = TRUE),
    meta = meta
  )
}

