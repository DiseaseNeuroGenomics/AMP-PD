from typing import Any, List, Optional

import os
import math
import pickle
import numpy as np
import scanpy as sc
from anndata.experimental.multi_files import AnnCollection


class CreateData:
    """Class to save the data in numpy memmap format
    Will save the gene expression using int16 to save space,
    but will create the function to normalize the data using
    the gene expression statistics"""

    def __init__(
        self,
        source_paths: List[str],
        target_path: str,
        obs_keys: List[str], # list of observations to include (e.g. Braak, Dementia, SubID, etc.)
        cell_restrictions: Optional[dict[Any]] = None, # can restrict which cell types to include
        min_genes_per_cell: int = 1000,
        protein_coding_only: bool = True,
    ):
        self.source_paths = source_paths
        self.target_path = target_path
        self.obs_keys = obs_keys
        self.cell_restrictions = cell_restrictions

        self.min_genes_per_cell = min_genes_per_cell
        self.protein_coding_only = protein_coding_only

        self.var_keys = [
            'gene_id', 'gene_name', 'gene_type', 'gene_chrom', 'ribosomal',
            'mitochondrial', 'protein_coding', 'percent_cells',
        ]

        if len(source_paths) == 1:
            self.anndata = sc.read_h5ad(source_paths[0], 'r')
        else:
            temp = [sc.read_h5ad(fn, 'r') for fn in source_paths]
            self._quality_check(temp)
            self.anndata = AnnCollection(temp, join_vars='inner', join_obs='inner')
            self.anndata.var = temp[0].var

        self._get_cell_index()
        self._get_gene_index()

        print(f"Size of anndata {self.anndata.shape[0]}")
        

    def _quality_check(self, data: List[Any]):
        """Ensure that the first two Anndata objects have matching gene names and percent cells"""
        vars = ["gene_name", "percent_cells"]
        for v in vars:
            match = [g0 == g1 for g0, g1 in zip(data[0].var[v], data[1].var[v])]
            assert np.mean(np.array(match)) == 1, f"{v} DID NOT MATCH match between the first two datasets"

            print(f"{v} matched between the first two datasets")


    def _create_metadata(self, train: bool = True):

        meta = {
            "obs": {k: self.anndata.obs[k][self.cell_idx].values for k in self.obs_keys},
            "var": {k: self.anndata.var[k][self.gene_idx].values for k in self.var_keys},
        }
        
        meta["obs"]["barcode"] = list(np.array(self.anndata.obs.index)[self.cell_idx])
        if not "SubID" in meta["obs"].keys() and "participant_id" in meta["obs"].keys():
            meta["obs"]["SubID"] = meta["obs"]["participant_id"] 

        meatadata_fn = os.path.join(self.target_path, "metadata.pkl")
        pickle.dump(meta, open(meatadata_fn, "wb"))


    def _get_gene_index(self):

        if self.protein_coding_only:
            self.gene_idx = np.where(self.anndata.var["protein_coding"])[0]
        else:
            self.gene_idx = np.arange(len(self.anndata.var["gene_name"]))

        print(f"Number of genes selected: {len(self.gene_idx)}")

    def _get_cell_index(self):

        cond = self.anndata.obs["n_genes"].values >= self.min_genes_per_cell
        if self.cell_restrictions is not None:
            for k, v in self.cell_restrictions.items():
                cond *= self.anndata.obs[k] == v

        self.cell_idx = np.where(cond)[0]
        print(f"Number of eligible cells: {len(self.cell_idx)}")

    def _create_gene_data(self):

        data_fn = os.path.join(self.target_path, "data.dat")
        n_genes = len(self.gene_idx)
        n_cells = len(self.cell_idx)
        print(f"Creating data. Number of cell: {n_cells}, number of genes: {n_genes}")

        chunk_size = 10_000  # chunk size for loading data into memory
        fp = np.memmap(data_fn, dtype='uint8', mode='w+', shape=(n_cells, n_genes))

        for n in range(math.ceil(n_cells / chunk_size)):
            m = np.minimum(n_cells, (n + 1) * chunk_size)
            current_idx = self.cell_idx[n * chunk_size: m]
            y = self.anndata[current_idx]
            y = y.X.toarray()
            y = y[:, self.gene_idx]
            y[y >= 255] = 255
            y = y.astype(np.uint8)
            fp[n * chunk_size: m, :] = y

            print(f"Chunk number {n} out of {int(np.ceil(len(self.cell_idx) / chunk_size))} created")

        # flush to memory
        fp.flush()

        return fp

    def create_dataset(self) -> None:

        print("Saving the metadata...")
        self._create_metadata()

        print("Saving the expression data in the memmap array...")
        fp = self._create_gene_data()

def create_dataset(
    source_paths: List[str], 
    target_path: str,
    obs_keys: List[str],
    cell_restrictions: Optional[dict[Any]] = None,
) -> None:

    c = CreateData(source_paths, target_path, obs_keys, cell_restrictions = cell_restrictions)
    c.create_dataset()


"""
 self.obs_keys = [
            'CERAD', 'BRAAK_AD', 'BRAAK_PD', 'Dementia', 'AD', 'class', 'subclass', 'subtype', 'ApoE_gt', "OCD",
            'Sex', 'Head_Injury', 'Vascular', 'Age', 'Epilepsy', 'Seizures', 'Tumor', 'PD', 'ALS', "FTD", "ADHD",
            'CDRScore', 'PMI', 'Cognitive_Resilience', 'Cognitive_and_Tau_Resilience', 'SubID',
            'snRNAseq_ID', 'SCZ', 'MDD', 'Brain_bank', "n_counts", "n_genes",
        ]
"""


