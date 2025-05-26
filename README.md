# AMP-PD Single-Cell Analysis Pipeline

This repository contains scripts and notebooks used for the analysis of single-nucleus RNA-sequencing (snRNA-seq) data in the AMP-PD cohort. The project includes data preprocessing, clustering, cell type composition modeling, GWAS scoring, differential expression, Hotspot module detection, and more — supporting our accompanying manuscript.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Preprocessing & Clustering](#preprocessing--clustering)
- [GWAS Scoring with scDRS](#gwas-scoring-with-scdrs)
- [Cell Type Compositional Analysis](#cell-type-compositional-analysis)
- [Differential Expression with dreamlet](#differential-expression-with-dreamlet)
- [Variance Partitioning](#variance-partitioning-of-braak-lb-pathology)
- [Hotspot Module Detection](#hotspot-analysis-of-microgliapvm-cells)
- [Installation & Requirements](#installation--requirements)

---

## Project Overview

The AMP-PD pipeline integrates multiple downstream single-cell analyses, including:

- Multilevel preprocessing (HVG, Harmony, Leiden)
- GWAS gene-set scoring (scDRS)
- Mixed model-based DE and variance partitioning (dreamlet)
- Cell composition analysis via pseudobulk modeling (crumblr)
- Hotspot-based gene module inference
- Cell-cell interaction and downstream network inference

---

## Preprocessing & Clustering

This script filters, normalizes, and integrates AMP-PD snRNA-seq data using Scanpy and Pegasus. It applies HVG selection, PCA, Harmony batch correction, and Leiden clustering.

### Run

```bash
python src/prepare_and_integrate_AMPPD.py
```

---

## GWAS Scoring with scDRS

This script scores individual cells for Parkinson’s disease GWAS gene sets using the SCDRS method.

### Run

```bash
python src/score_gwas_scdrs.py
```

---

## Cell Type Compositional Analysis

This module uses `crumblr` to analyze changes in subclass-level cell type proportions across Braak stages and diagnosis groups using pseudobulk counts.

### Script

- `cell_type_composition.R`

### Highlights

- Converts cell count matrices via `crumblr`
- Runs mixed models with `dream` on cell proportions
- Performs meta-analysis with `metafor`
- Visualizes subclass shifts on hierarchical tree (`ggtree`)

---

## Differential Expression with dreamlet

This script performs differential gene expression modeling across pseudobulk samples using `dreamlet`, adjusting for biological and technical covariates.

### Script

- `dreamlet_differential_expression_PD.R`

### Outputs

- DE result tables (`.csv`)
- Model objects (`.RData`)

---

## Variance Partitioning of Braak LB Pathology

This script quantifies sources of variation (biological and technical) in gene expression across AMP-PD samples using `dreamlet::fitVarPart()`.

### Run

```bash
Rscript scripts/variance_partition_braakLB_dreamlet.R
```

---

## Hotspot Analysis of Myeloid Cells

This script applies `Hotspot` to identify locally co-regulated gene modules within microglia/PVM cells.

### Workflow

- Load Harmony-integrated AnnData file
- Match with updated cell metadata
- Run autocorrelation + module discovery
- Export scores and modules

### Run

```bash
python src/hotspot_micro_pvm.py
```

---


## CCI calling

This script allows to infer CCIs.

### Run

```bash
python src/run_cci_liana_example.py
```



## Installation & Requirements

Most scripts require the following R and Python packages (via CRAN, Bioconductor, or pip):

### Python (pip)

```bash
pip install scanpy pegasusio pegasuspy scdrs matplotlib seaborn pandas hotspotsc seaborn harmony anndata 
```

### R (Bioconductor + CRAN)

```r
# Bioconductor
BiocManager::install(c(
  "zellkonverter", "SingleCellExperiment", "dreamlet", "variancePartition",
  "crumblr", "ggtree", "qvalue", "GSEABase", "BiocParallel"
))

# CRAN
install.packages(c(
  "ggplot2", "tidyverse", "aplot", "broom", "cowplot", "metafor", "reticulate"
))
```
