# AMP-PD Single-Cell Analysis Pipeline

This repository contains scripts and notebooks used for the analysis of single-nucleus RNA-sequencing (snRNA-seq) data in the AMP-PD cohort. The project includes data preprocessing, clustering, cell type composition modeling, GWAS scoring, differential expression, TF inference, Hotspot module detection, and more — supporting our accompanying manuscript.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Preprocessing & Clustering](#preprocessing--clustering)
- [GWAS Scoring with scDRS](#gwas-scoring-with-scdrs)
- [Cell Type Compositional Analysis](#cell-type-compositional-analysis)
- [Differential Expression with dreamlet](#differential-expression-with-dreamlet)
- [Variance Partitioning](#variance-partitioning)
- [Hotspot Module Detection](#hotspot-module-detection)
- [TF Activity Inference](#tf-activity-inference)
- [Cell-Cell Interaction Inference](#cell-cell-interaction-inference)
- [Installation & Requirements](#installation--requirements)

---

## Project Overview

This pipeline supports the full spectrum of single-cell and pseudobulk-level analyses in the AMP-PD dataset. Modules include:

- Multistage data preprocessing, QC, and integration (Scanpy + Harmony)
- Cell-type proportion modeling across clinical groups (crumblr + dream)
- GWAS signal scoring with scDRS
- Differential expression via dreamlet with hierarchical covariates
- Hotspot-based gene module discovery
- TF activity inference and regulatory network visualization (decoupler)
- Cell-cell interaction prediction (LIANA)

---

## Preprocessing & Clustering

**Script:** `prepare_and_integrate_AMPPD.py`  
Performs HVG filtering, PCA, Harmony batch correction, and Leiden clustering using Scanpy and Pegasus.

### Run

```bash
python src/prepare_and_integrate_AMPPD.py
```

---

## GWAS Scoring with scDRS

**Script:** `score_gwas_scdrs.py`  
Scores single cells using scDRS for prioritized Parkinson’s disease GWAS gene sets.

### Run

```bash
python src/score_gwas_scdrs.py
```

---

## Cell Type Compositional Analysis

**Script:** `cell_type_composition.R`  
Performs compositional modeling across subclasses using `crumblr` and `dream`, followed by meta-analysis with `metafor` and visualization via `ggtree`.

### Output

- Meta-analysis of compositional shifts
- Coefficient plots annotated on hierarchical cell tree

---

## Differential Expression with dreamlet

**Script:** `dreamlet_differential_expression_PD.R`  
Runs subclass-level pseudobulk DE using `dreamlet`. Includes models with and without ethnicity/participant covariates.

### Output

- DE results in `.csv` and `.RData`

---

## Variance Partitioning

**Script:** `variance_partition_braakLB_dreamlet.R`  
Uses `dreamlet::fitVarPart()` to quantify gene expression variance explained by covariates including Braak LB stage.

### Run

```bash
Rscript scripts/variance_partition_braakLB_dreamlet.R
```

---

## Hotspot Module Detection

**Script:** `hotspot_example.py`  
Runs the Hotspot algorithm to detect local autocorrelated gene modules, focusing on myeloid cells.

### Run

```bash
python src/hotspot_example.py
```

---

## TF Activity Inference

**Script:** `tf_inference.py`  
Infers transcription factor activity per cell type using `decoupler` and CollecTRI network. Includes specificity scoring, normalized activity, and heatmap visualization.

### Output

- Normalized TF activity scores
- TF specificity scores
- Ranked TFs per subclass with heatmap

---

## Cell-Cell Interaction Inference

**Script:** `run_cci_liana_example.py`  
Uses the `LIANA` Python package to infer cell-cell interactions from integrated single-cell data.

### Run

```bash
python src/run_cci_liana_example.py
```

---

## Installation & Requirements

### Python (via pip)

The following packages are required and can be installed via pip:

```bash
pip install scanpy pegasuspy pegasusio anndata scdrs matplotlib seaborn pandas numpy \
    decoupler liana igraph scikit-learn harmony-pytorch
```

Note:
- `pegasuspy` and `pegasusio` are required for HDF5/AnnData processing
- `harmony-pytorch` is used for batch correction
- `decoupler` is used for TF activity inference
- `liana` is used for CCI prediction

---

### R (via Bioconductor + CRAN)

Use the following commands to install the required R packages:

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

---


