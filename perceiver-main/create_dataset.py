from typing import Any, List, Optional

import os
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
        train_pct: float = 0.9,
        min_genes_per_cell: int = 1000,
        min_percent_cells_per_gene: float = 0.02,
        split_train_test_by_subject: bool = True,
        protein_coding_only: bool = True,
    ):
        self.source_paths = source_paths
        self.target_path = target_path
        self.train_pct = train_pct

        self.min_genes_per_cell = min_genes_per_cell
        self.min_percent_cells_per_gene = min_percent_cells_per_gene
        self.split_train_test_by_subject = split_train_test_by_subject
        self.protein_coding_only = protein_coding_only

        self.obs_keys = [
            'CERAD', 'BRAAK_AD', 'BRAAK_PD', 'Dementia', 'AD', 'class', 'subclass', 'subtype', 'ApoE_gt',
            'Sex', 'Head_Injury', 'Vascular', 'Age', 'Epilepsy', 'Seizures', 'Tumor', 'PD', 'ALS',
            'CDRScore', 'PMI', 'Cognitive_Resilience', 'Cognitive_and_Tau_Resilience', 'SubID',
            'snRNAseq_ID', 'SCZ', 'MDD', 'Brain_bank',
        ]
        self.var_keys = [
            'gene_id', 'gene_name', 'gene_type',  'robust', 'highly_variable_features', 'ribosomal',
            'mitochondrial', 'protein_coding', 'mitocarta_genes', 'robust_protein_coding', 'percent_cells',
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

    def _train_test_splits(self):

        if self.split_train_test_by_subject:
            sub_ids = np.unique(self.anndata.obs["SubID"].values)
            np.random.shuffle(sub_ids)
            n = len(sub_ids)
            train_ids = sub_ids[: int(n * self.train_pct)]
            test_ids = sub_ids[int(n * self.train_pct):]
            self.train_idx = [n for n, s_id in enumerate(self.anndata.obs["SubID"].values) if s_id in train_ids]
            self.test_idx = [n for n, s_id in enumerate(self.anndata.obs["SubID"].values) if s_id in test_ids]
            print(
                f"Splitting the train/test set by SubID. "
                f"{len(train_ids)} subjects in train set; {len(test_ids)} subjects in test set"
            )
        else:
            np.random.shuffle(self.cell_idx)
            n = len(self.cell_idx)
            self.train_idx = self.cell_idx[: int(n * self.train_pct)]
            self.test_idx = self.cell_idx[int(n * self.train_pct):]
            print(
                f"Randomly splitting the train/test. {len(self.train_idx)} cells in train set; "
                f"{len(self.test_idx)} cells in test set"
            )

        # sorting for more efficient reading from AnnData (I think ...)
        self.train_idx = np.sort(self.train_idx)
        self.test_idx = np.sort(self.test_idx)

    def _create_metadata(self, train: bool = True):

        idx = self.train_idx if train else self.test_idx
        meatadata_fn = "train_metadata.pkl" if train else "test_metadata.pkl"
        meatadata_fn = os.path.join(self.target_path, meatadata_fn)

        meta = {
            "obs": {k: self.anndata.obs[k][idx].values for k in self.obs_keys},
            "var": {k: self.anndata.var[k][self.gene_idx].values for k in self.var_keys},
        }

        pickle.dump(meta, open(meatadata_fn, "wb"))

    def _get_gene_index(self):

        if "percent_cells" in self.anndata.var.keys():
            self.gene_idx = self.anndata.var["percent_cells"] > self.min_percent_of_cells
            if self.protein_coding_only:
                self.gene_idx *= self.anndata.var["protein_coding"]

            self.gene_idx = np.where(self.gene_idx)[0]

        else:
            chunk_size = 10_000
            n_segments = 10
            n = self.anndata.shape[0]
            start_idx = np.linspace(0, n - chunk_size - 1, n_segments)
            gene_expression = []

            for i in start_idx:
                x = self.anndata[int(i): int(i + chunk_size)]
                x = x.X.toarray()
                gene_expression.append(np.mean(x > 0, axis=0))

            gene_expression = np.mean(np.stack(gene_expression), axis=0)
            self.gene_idx = np.where(gene_expression >= self.min_percent_cells_per_gene)[0]

        print(f"Number of genes selected: {len(self.gene_idx)}")

    def _get_cell_index(self):

        n_genes = self.anndata.obs["n_genes"].values
        self.cell_idx = np.where(n_genes > self.min_genes_per_cell)[0]

    def _create_dataset(self, train: bool = True):

        idx = self.train_idx if train else self.test_idx
        data_fn = "train_data.dat" if train else "test_data.dat"
        data_fn = os.path.join(self.target_path, data_fn)
        n_genes = len(self.gene_idx)
        print(f"Creating data. Number of cell: {len(idx)}, number of genes: {n_genes}")

        chunk_size = 10_000  # chunk size for loading data into memory
        fp = np.memmap(data_fn, dtype='unint8', mode='w+', shape=(len(idx), n_genes))

        for n in range(int(np.ceil(len(idx) / chunk_size))):
            m = np.minimum(len(idx), (n + 1) * chunk_size)
            current_idx = idx[n * chunk_size: m]
            print(f"Creating dataset, cell number = {current_idx[0]}")
            y = self.anndata[current_idx]
            y = y.X.toarray()
            y = y[:, self.gene_idx]
            y[y >= 255] = 255
            y = y.astype(np.uint8)
            fp[n * chunk_size: m, :] = y

            print(f"Chunk number {n} out of {int(np.ceil(len(idx) / chunk_size))} created")

        # flush to memory
        fp.flush()

        return fp

    def create_datasets(self) -> None:

        np.random.seed(seed=42)

        # randomly choose the train/test splits
        self._train_test_splits()

        print("Saving the training data in the memmap array...")
        # must create the training set first, since gene stats are calculated using
        fp = self._create_dataset(train=True)

        #print("Calculating the gene expression stats...")
        #self._calculate_stats(fp)

        print("Saving the training metadata...")
        self._create_metadata(train=True)

        print("Saving the test data in the memmap array...")
        _ = self._create_dataset(train=False)

        print("Saving the test metadata...")
        self._create_metadata(train=False)


if __name__ == "__main__":

    base_dir = "/sc/arion/projects/psychAD/NPS-AD/freeze2_rc/h5ad_final/"
    source_paths = [
        base_dir + "RUSH_2023-06-08_21_44.h5ad",
        base_dir + "MSSM_2023-06-08_22_31.h5ad",
    ]
    target_path = "/sc/arion/projects/psychAD/massen06/mssm_rush_data"
    c = CreateData(source_paths, target_path, train_pct=0.9)

    c.create_datasets()


