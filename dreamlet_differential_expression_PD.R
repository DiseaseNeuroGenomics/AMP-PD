#!/usr/bin/env Rscript

# ==========================================================
# Script: dreamlet_differential_expression_PD.R
# Author: Tereza Clarence
# Date:   2025-05-13
#
# Description:
# ------------
# This script runs differential expression analysis using `dreamlet`
# on pre-processed AMP-PD pseudobulk data across subclass resolution.
# It includes models with and without covariates like ethnicity.
#
# Outputs:
# - DE results saved as .RData and .csv
#
# Requirements:
# - dreamlet >= v1.0.0
# - crumbler, zenith, qvalue, GSEABase
# - BiocParallel (for multithreading)
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

cat("Working directory:\n")
print(getwd())


# --- Load Processed Data ---

load("../VarPart_resproc_CLASS.RData")
# This loads: res.proc_subclass2 (processed pseudobulk data object)


# --- Define Common Settings ---

CONTRASTS <- c(Diff = "dxPD - dxCTRL")
COEF <- "Diff"


# --- Model 1: Baseline with main covariates only ---

form_dreamlet1 <- ~ scale(age) +
  (1 | brain_reg_w_gt) + (1 | sex) + dx + (1 | Source) +
  RIN + log(n_genes) + percent_mito + ribo_genes + mito_ribo + 0

res_dl_dx1 <- dreamlet(res.proc_subclass2, form_dreamlet1, contrasts = CONTRASTS)

save(form_dreamlet1, res_dl_dx1,
     file = "Resdl_CLASS_base_DiffPDvsCTRL_raw.RData")

write.csv(topTable(res_dl_dx1, coef = COEF, number = Inf),
          "./topTable_dxPD__CLASS_base_DiffPDvsCTRL_raw.csv")


# --- Model 2: Fully adjusted model including ethnicity and participant ID ---

form_dreamlet2 <- ~ scale(age) +
  (1 | brain_reg_w_gt) + (1 | sex) + dx + (1 | Source) +
  (1 | Ethnicity) + (1 | participant_id) +
  RIN + log(n_genes) + percent_mito + ribo_genes + mito_ribo + 0

res_dl_dx2 <- dreamlet(res.proc_subclass2, form_dreamlet2, contrasts = CONTRASTS)

save(form_dreamlet2, res_dl_dx2,
     file = "Resdl_CLASS_base_DiffPDvsCTRL_Ethnicity_raw.RData")

write.csv(topTable(res_dl_dx2, coef = COEF, number = Inf),
          "./topTable_dxPD__CLASS_base_DiffPDvsCTRL_Ethnicity_raw.csv")



# --- Optional: Summarize DE Results ---

# summary_tab <- topTable(res_dl_dx2, coef = COEF, number = Inf) %>%
#   as_tibble() %>%
#   group_by(assay) %>%
#   summarize(
#     nDE = sum(adj.P.Val < 0.05),
#     pi1 = 1 - pi0est(P.Value)$pi0,
#     nGenes = length(adj.P.Val)
#   ) %>%
#   mutate(assay = factor(assay, assayNames(res_dl_dx2)))
#
# write.csv(summary_tab, "./summary_dxPD_CLASS_cova_DiffPDvsCTRL_raw.csv")
