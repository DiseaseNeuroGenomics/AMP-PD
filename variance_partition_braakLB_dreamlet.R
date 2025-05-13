#!/usr/bin/env Rscript
# ==========================================================
# Script: variance_partition_braakLB_dreamlet.R
# Author: Tereza Clarence
# Date:   2025-05-13
#
# Description:
# ------------
# This script performs variancePartition modeling of Braak LB pathology
# using AMP-PD single-cell RNA-seq pseudobulk data aggregated at the
# 'class' level. Metadata is merged and covariates are derived to model
# cellular transcriptomic variance across Braak LB stages.
#
# Output:
# - Processed dreamlet object (voom + weights)
# - Fitted variance partitioning model
#
# Requirements:
# - dreamlet, zellkonverter, variancePartition, BiocParallel
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
})

# --- Print Versions for Reproducibility ---

cat("Session package versions:\n")
cat(paste0("dreamlet v", packageVersion("dreamlet")), "\n")
cat(paste0("crumblr v", packageVersion("crumblr")), "\n")
cat(paste0("variancePartition v", packageVersion("variancePartition")), "\n")
cat(paste0("zenith v", packageVersion("zenith")), "\n")
cat(paste0("zellkonverter v", packageVersion("zellkonverter")), "\n")
cat(paste0("BiocManager v", BiocManager::version()), "\n")

print(getwd())


# --- Step 1: Load H5AD and Metadata ---

h5ad_file <- "./AMP_PD.h5ad"
sce <- readH5AD(h5ad_file, use_hdf5 = TRUE)
print(sce)

df_meta <- read.csv('./donor_metadata.csv')
df_meta_filtered <- df_meta[df_meta$participant_id %in% sce$participant_id, ]
df_meta_ordered <- df_meta_filtered[match(sce$participant_id, df_meta_filtered$participant_id), ]

# Sanity check
stopifnot(nrow(df_meta_ordered) == ncol(sce))


# --- Step 2: Add Covariates to SCE ---

sce$PMI <- df_meta_ordered$PMI.PMI_hours
sce$Ethnicity <- df_meta_ordered$Demographics.ethnicity
sce$project <- 'AMP-PD'
sce$sample_id <- paste(sce$participant_id, sce$brain_reg_w_gt, sep = "_")
sce$dx <- sce$diagnosis_final

# Braak LB group: Low (0–2), Moderate (3–4), High (5–6)
sce$braak_lb_group <- cut(sce$path_braak_lb,
                          breaks = c(-1, 2, 4, 6),
                          labels = c("Low", "Moderate", "High"),
                          ordered_result = TRUE)

# Braak LB stage: Control, Early, Medium, Late
sce$braak_lb_stage <- ifelse(
  sce$path_braak_lb >= 1 & sce$path_braak_lb <= 2, "Early",
  ifelse(sce$path_braak_lb >= 3 & sce$path_braak_lb <= 4, "Medium",
         ifelse(sce$path_braak_lb >= 5 & sce$path_braak_lb <= 6, "Late", "Control"))
)
sce$braak_lb_stage <- factor(sce$braak_lb_stage, levels = c("Control", "Early", "Medium", "Late"))


# --- Step 3: Pseudobulk Aggregation by Class ---

pbSC <- aggregateToPseudoBulk(
  sce,
  assay = "X", # assumes counts stored in X
  cluster_id = "class",
  sample_id = "sample_id",
  BPPARAM = SnowParam(workers = 6, progressbar = TRUE)
)
print(pbSC)


# --- Step 4: Stack Assays Across Classes ---

pb_stack <- stackAssays(pbSC)
print(pb_stack)


# --- Step 5: Define Model and Process with dreamlet ---

form_varpart <- ~ scale(age) +
  (1 | brain_reg_w_gt) + (1 | stackedAssay) +
  (1 | sex) + path_braak_lb + (1 | diagnosis_final) +
  (1 | Ethnicity) + (1 | Source) + (1 | participant_id) +
  RIN + log(n_genes) + PMI + percent_mito + ribo_genes +
  mito_ribo + log(n_counts)

# Normalize, filter, voom + weights
res_proc_class <- processAssays(
  pb_stack,
  form_varpart,
  min.count = 5
)

save(form_varpart, res_proc_class,
     file = "stackedVarPart_resproc_CLASS_raw.RData")


# --- Step 6: Fit Variance Partition Model ---

vp_lst <- fitVarPart(res_proc_class, form_varpart)
save(form_varpart, vp_lst,
     file = "stackedVarPartFit_CLASS_raw.RData")

# Optional: plotVarPart(vp_lst, label.angle = 60)
