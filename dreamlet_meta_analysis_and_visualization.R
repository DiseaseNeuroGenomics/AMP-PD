#!/usr/bin/env Rscript
# ==========================================================
# Script: dreamlet_meta_analysis_and_visualization.R
# Author: Tereza Clarence
# Date:   2025-05-13
#
# Description:
# ------------
# This script runs multiple differential expression models using `dream` from
# `variancePartition`, aggregates contrasts using meta-analysis with `metafor`,
# and visualizes effect sizes across subclass-level cell types in AMP-PD data.
#
# Output:
# - Meta-analysis DE results per contrast
# - Annotated coefficient plots with phylogenetic cell tree
#
# Requirements:
# - dreamlet, crumbler, metafor, ggtree, broom, tidyverse
# ==========================================================

# --- Load Libraries ---
suppressPackageStartupMessages({
  library(zellkonverter)
  library(SingleCellExperiment)
  library(dreamlet)
  library(ggplot2)
  library(tidyverse)
  library(aplot)
  library(ggtree)
  library(scattermore)
  library(zenith)
  library(crumblr)
  library(GSEABase)
  library(qvalue)
  library(BiocParallel)
  library(cowplot)
  library(DelayedArray)
  library(broom)
  library(reticulate)
  library(metafor)
})

# --- Print Package Versions ---
cat("Session package versions:\n")
cat(paste0("dreamlet v", packageVersion("dreamlet")), "\n")
cat(paste0("crumblr v", packageVersion("crumblr")), "\n")
cat(paste0("variancePartition v", packageVersion("variancePartition")), "\n")
cat(paste0("zenith v", packageVersion("zenith")), "\n")
cat(paste0("zellkonverter v", packageVersion("zellkonverter")), "\n")
cat(paste0("BiocManager v", BiocManager::version()), "\n")

# --- Utility Functions ---
meta_analysis <- function(tabList) {
  if (is.null(names(tabList))) names(tabList) <- as.character(seq_along(tabList))
  for (key in names(tabList)) tabList[[key]]$Dataset <- key
  df <- bind_rows(tabList)
  df %>% group_by(assay) %>% do(tidy(rma(yi = logFC, sei = logFC / t, data = ., method = "FE"))) %>%
    ungroup() %>% mutate(FDR = p.adjust(p.value, "fdr"), log10FDR = -log10(FDR))
}

plotTree <- function(tree, low = "grey90", mid = "red", high = "darkred", xmax.scale = 1.5) {
  fig <- ggtree(tree, branch.length = "none") +
    geom_tiplab(color = "black", size = 4, hjust = 0, offset = .2) +
    theme(legend.position = "top left", plot.title = element_text(hjust = 0.5))
  xmax <- layer_scales(fig)$x$range$range[2]
  fig + xlim(0, xmax * xmax.scale)
}

plotCoef2 <- function(tab, coef, fig.tree, low = "grey90", mid = "red", high = "darkred", ylab, tick_size) {
  tab <- tab %>% mutate(celltype = factor(assay, levels = rev(get_taxa_name(fig.tree))))
  ggplot(tab, aes(celltype, estimate)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey", linewidth = 1) +
    geom_errorbar(aes(ymin = estimate - 1.96 * std.error, ymax = estimate + 1.96 * std.error), width = 0) +
    geom_point2(aes(color = pmin(4, -log10(FDR)), size = pmin(4, -log10(FDR)))) +
    scale_color_gradient2(name = bquote(-log[10]~FDR), limits = c(0, 4), low = low, mid = mid, high = high, midpoint = -log10(0.01)) +
    scale_size_area(name = bquote(-log[10]~FDR), limits = c(0, 4)) +
    geom_text2(aes(label = '+', subset = FDR < 0.05), color = "white", size = 6, vjust = .3, hjust = .5) +
    theme_classic() + coord_flip() +
    xlab('') + ylab(ylab) +
    theme(axis.text.y = element_blank(), axis.text = element_text(size = 12),
          axis.ticks.y = element_blank(), text = element_text(size = tick_size)) +
    scale_y_continuous(breaks = scales::breaks_pretty(3))
}

coef_prep <- function(df) {
  df %>% group_by(assay) %>% do(tidy(rma(yi = logFC, sei = logFC / t, data = ., method = "FE"))) %>%
    ungroup() %>% mutate(FDR = p.adjust(p.value, "fdr"), log10FDR = -log10(FDR))
}



# --- Load Data ---
FPath <- "./"
file <- 'AMP_PD.h5ad'
sce <- readH5AD(file.path(FPath, file), use_hdf5 = TRUE, raw = FALSE)

# --- Add Metadata ---
df_meta <- read.csv('./donor_metadata.csv')
df_meta_ordered <- df_meta[match(sce$participant_id, df_meta$participant_id), ]
sce$PMI <- df_meta_ordered$PMI.PMI_hours
sce$Ethnicity <- df_meta_ordered$Demographics.ethnicity
sce$sample_id <- paste(sce$participant_id, sce$brain_reg_w_gt, sep = "_")
sce$braak_lb_group <- cut(sce$path_braak_lb, breaks = c(-1, 2, 4, 6), labels = c("Low", "Moderate", "High"), ordered_result = TRUE)
sce$braak_lb_stage <- factor(ifelse(sce$path_braak_lb >= 1 & sce$path_braak_lb <= 2, "Early",
                              ifelse(sce$path_braak_lb >= 3 & sce$path_braak_lb <= 4, "Moderate",
                                     ifelse(sce$path_braak_lb >= 5 & sce$path_braak_lb <= 6, "Late", "Control"))),
                              levels = c("Control", "Early", "Moderate", "Late"))

# --- Aggregate to Pseudobulk ---
pb <- aggregateToPseudoBulk(sce, assay = "X", cluster_id = "derived_subclass2_Dec2024", sample_id = "sample_id", BPPARAM = SnowParam(6))
metadata(pb)$aggr_means %>% group_by(sample_id) %>% summarize(across(c(n_counts, n_genes, percent_mito, mito_ribo), mean, na.rm = TRUE)) -> nc
colData(pb) <- merge(colData(pb), nc[match(nc$sample_id, colData(pb)$sample_id), ], by = 'sample_id', all.x = TRUE)
rownames(colData(pb)) <- colData(pb)$sample_id
PB0 <- pb
cobj <- crumblr(cellCounts(PB0))

# --- Run Dream Models and Extract Coefficients ---
form <- ~ scale(age) + (1|brain_reg_w_gt) + (1|Source) + dxNEW + RIN + (1|sex) + scale(percent_mito) + scale(n_counts) + 0
L <- makeContrastsDream(form, colData(PB0), contrasts = c(
  dxDiffLateCtrl = "dxNEWLate - dxNEWControl",
  dxDiffEarlyCtrl = "dxNEWEarly - dxNEWControl",
  dxDiffMediumCtrl = "dxNEWModerate - dxNEWControl",
  dxDiffLateEarly = "dxNEWLate - dxNEWEarly",
  dxDiffEarlyLate = "dxNEWEarly - dxNEWLate"
))
fit <- dream(cobj, form, colData(PB0), L = L)
fit0 <- eBayes(fit)

# --- Meta-analysis per Contrast ---
coef_prep <- function(df) {
  df %>% group_by(assay) %>% do(tidy(rma(yi = logFC, sei = logFC / t, data = ., method = "FE"))) %>%
    ungroup() %>% mutate(FDR = p.adjust(p.value, "fdr"), log10FDR = -log10(FDR))
}

res.earlyctrl <- coef_prep(topTable(fit0, coef = 'dxDiffEarlyCtrl', number = Inf) %>% rownames_to_column('assay'))
res.medctrl <- coef_prep(topTable(fit0, coef = 'dxDiffMediumCtrl', number = Inf) %>% rownames_to_column('assay'))
res.latectrl <- coef_prep(topTable(fit0, coef = 'dxDiffLateCtrl', number = Inf) %>% rownames_to_column('assay'))
res.earlylate <- coef_prep(topTable(fit0, coef = 'dxDiffEarlyLate', number = Inf) %>% rownames_to_column('assay'))
res.lateearly <- coef_prep(topTable(fit0, coef = 'dxDiffLateEarly', number = Inf) %>% rownames_to_column('assay'))

# --- Plot Tree and Coefficients ---
hc <- buildClusterTreeFromPB(PB0)
fig.tree <- plotTree(ape::as.phylo(hc), xmax.scale = 2.2) + theme(legend.position = "bottom")

fig.es1 <- plotCoef2(res.earlyctrl, coef = 'dxDiffEarlyCtrl', fig.tree, ylab = 'Early vs Ctrl', tick_size = 10)
fig.es2 <- plotCoef2(res.medctrl, coef = 'dxDiffMediumCtrl', fig.tree, ylab = 'Medium vs Ctrl', tick_size = 10)
fig.es3 <- plotCoef2(res.latectrl, coef = 'dxDiffLateCtrl', fig.tree, ylab = 'Late vs Ctrl', tick_size = 10)
fig.es4 <- plotCoef2(res.braak, coef = 'path_braak_lb', fig.tree, ylab = 'Braak LB', tick_size = 10)

options(repr.plot.width = 19, repr.plot.height = 8)
fig.es1 %>% insert_left(fig.tree, width = 1.5) %>%
  insert_right(fig.es2, width = 1.0) %>%
  insert_right(fig.es3, width = 1.0) %>%
  insert_right(fig.es4, width = 1.0)


# End of script
