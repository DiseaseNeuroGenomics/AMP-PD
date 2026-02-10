# Disease Similarity Score (DSS)

This repository implements the Disease Similarity Score (DSS), a correlation-based
framework for quantifying donor-level similarity to disease-associated transcriptional
signatures using pseudobulk single-cell RNA-seq data.

## Key features
- Cell-type–resolved scoring
- Control-referenced baseline correction
- Disease-weighted gene correlations
- Robust to scale and moderate cell-count variability

## Outputs
- DSS matrix (cell type × donor)
- Donor-level average DSS
- Metadata-aligned results for downstream analysis

## Requirements
- R >= 4.2
- SingleCellExperiment
- matrixStats

