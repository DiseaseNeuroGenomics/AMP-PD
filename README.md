# AMP-PD


This repository contains scripts and notebooks used for single-cell RNA-sequencing (scRNA-seq) analysis as part of our manuscript. It includes data preprocessing, clustering, differential expression, and downstream analyses.

## Table of Contents

- [Project Overview](#project-overview)
- [Preprocessing & Clustering](#preprocess-cluster)
- [scDRS Calculation](#scDRS)
- [Cell-type Compositional Analysis](#crumblr)
- [Differential Expression Analysis](#dreamlet) 
- [Hotspot analysis](#hotspot)
- [Cell-Cell Interactions](#CCIs)
- [Installation & Requirements](#installation--requirements)



# GWAS Scoring with SCDRS

This script scores single-cell transcriptomes using SCDRS for Parkinson’s disease gene sets.

## Run

```bash
python src/score_gwas_scdrs.py




# AMP-PD Preprocessing Pipeline

This script filters, normalizes, and integrates AMP-PD single-nucleus RNA-seq data.
It applies HVG selection, PCA, Harmony batch correction, and Leiden clustering.

## Steps

- Subset to protein-coding autosomal genes
- Select top HVGs using `scanpy` method
- Regress out confounders, run Harmony integration
- UMAP and clustering

## Run

```bash
python src/prepare_and_integrate_AMPPD.py



# DREAMlet-based Differential Expression in AMP-PD

This repository contains differential gene expression scripts using `dreamlet`
on AMP-PD pseudobulk data. Models include core covariates and adjust for
individual-level and technical variation.

## Scripts

- `dreamlet_differential_expression_PD.R`: main DE script with 3 model variations

## Dependencies

See the top of the script for required R packages (Bioconductor + CRAN).

## Output

- `.csv`: differential expression results
- `.RData`: model objects and formulas


# Hotspot Analysis of Microglia/PVM Cells

This script performs Hotspot module detection in AMP-PD microglia/PVM cells after
batch correction with Harmony.

## Workflow

- Load processed AnnData file and updated metadata
- Match and annotate cells
- Filter genes and compute Hotspot autocorrelations
- Derive modules and export scores

## Run

```bash
python src/hotspot_micro_pvm.py

