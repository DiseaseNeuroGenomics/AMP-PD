# AMP-PD


This repository contains scripts and notebooks used for single-cell RNA-sequencing (scRNA-seq) analysis as part of our manuscript. It includes data preprocessing, clustering, differential expression, and downstream analyses.

## Table of Contents

- [Project Overview](#project-overview)
- [Preprocessing & Clustering](#preprocess-cluster)
- [scDRS Calculation](#scDRS)
- [Cell-type Compositional Analysis](#crumblr)
- [Differential Expression Analysis](#dreamlet)
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


- 
- [Hotspot analysis](#hotspot)
- [Cell-Cell Interactions](#CCIs)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Contact](#contact)


