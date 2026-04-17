# Trajectory analysis for AMPPD

### Requirements  
PyTorch and PyTorch Lightning (https://lightning.ai/docs/pytorch/stable/).  
Conda environment used for model training given in env.yaml.  
To run the R script zenith_loop.R, the libraries zenith and GSEABase are required.

### Step 1 - Create the dataset used for model training and analysis 
  
The worksheet create_ad_pd_datasets_worksheet.ipynb provides the steps required to generate the data required for model 
training and analysis of both the PD and AD datasets. The input to this pipeline is the experiment h5ad file, containing 
the gene expression data, and cell observation data, and the gene names. For this analysis, only myeloid/immune cells are included, 
but this can be modified.  
  
For each of PD and AD, three different files are created: 1) a numpy memmap .dat file which stores the gene expression data, a .pkl file which contains the cell and gene metadata, and a .pkl file containing the 20 train/test splits, in which donors only appear in the train or the test set. Given the size of these files, they are not included in the repo. All other intermeidate analysis files are.

### Step 2 - Train model and run inference  
Ensure the path names for gene data (.dat file), metadata (.pkl file) and train/test splits (.pkl) in config.py are correct. Feel free to modify any other network and training parameters.  
  
To train the model, run 
```
python train.py  
```
Models will be trained on all train/test splits. Model inference (e.g. predicted Braak scoress, cell index) will be saved in the log_path specified in config.py.
  
### Step 3 - Create AnnData structure (h5ad) with model predictions
In process_model.ipynb, we walk through the step required to create the h5ad files needed for downstream analysis. 

### Step 4 - Analysis
Analysis code is found in analysis_worksheet.py. Pathway enrichment requires running the R script zenith_loop.R found in R_scripts. Intermediate files are already provided.





