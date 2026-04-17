from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import os
import copy
import torch
import numpy as np
import scanpy as sc
import pickle
import pytorch_lightning as pl
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    BatchSampler,
    SequentialSampler,
    Subset,
)


class SingleCellDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        cell_idx: Optional[List[int]] = None,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        n_targets: Optional[int] = None,
        n_input: int = 100,
        batch_size: int = 32,
        normalize_total: bool = False,
        log_normalize: bool = True,
        random_scaling: bool = False,
        cell_prop_same_ids: bool = False,
        max_cell_prop_val: float = 999,
        protein_coding_only: bool = False,
        remove_sex_chrom: bool = False,
        n_bins: Optional[int] = None,
        embedding_strategy: Literal["binned", "continuous", "film"] = "continuous",
        n_genes_per_input: int = 400,
        cell_restrictions: Optional[Dict[str, Any]] = None,
        RDA: bool = False, # read-depth-aware, from scFoundation
        masked_gene_selection: Literal["BERT", "exclusive", "non_exclusive", "VAE"] = "BERT",
        exclude_gene_val: int = 255,
        training: bool = True,
    ):

        self.metadata = pickle.load(open(metadata_path, "rb"))
        if not "barcode" in self.metadata["obs"]:
            self.metadata["obs"]["barcode"] = np.arange(len(self.metadata["obs"]["experiment"]))
        self.data_path = data_path
        self.cell_idx = cell_idx if cell_idx is not None else np.arange(len(self.metadata["obs"]["barcode"]))
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties

        self.n_samples = len(self.metadata["obs"]["barcode"])
        self.cell_restrictions = cell_restrictions

        self._restrict_samples(cell_restrictions)

        print(f"Number of cells {self.n_samples}")
        if "gene_name" in self.metadata["var"].keys():
            self.n_genes_original = len(self.metadata["var"]["gene_name"])
        else:
            self.n_genes_original = len(self.metadata["var"])
        self.n_cell_properties = len(cell_properties) if cell_properties is not None else 0
        self.n_targets = n_targets
        self.n_input = n_input
        self.batch_size = batch_size

        self.normalize_total = normalize_total
        self.log_normalize = log_normalize
        self.random_scaling = random_scaling
        self.embedding_strategy = embedding_strategy
        self.n_bins = n_bins
        self.cell_prop_same_ids = cell_prop_same_ids
        self.max_cell_prop_val = max_cell_prop_val
        self.protein_coding_only = protein_coding_only
        self.remove_sex_chrom = remove_sex_chrom
        self.n_genes_per_input = n_genes_per_input
        self.RDA = RDA
        self.masked_gene_selection = masked_gene_selection
        self.exclude_gene_val = exclude_gene_val
        self.training = training

        self.offset = 1 * self.n_genes_original  # UINT8 is 1 bytes

        self._get_gene_index()
        self._create_gene_cell_prop_ids()

        self._get_cell_prop_vals()
        self._get_batch_prop_vals()

        if embedding_strategy == "binned":
            self._define_bins()


    def __len__(self):
        return self.n_samples

    def _define_bins(self):

        self.bins = [-1]
        while len(self.bins) < self.n_bins:
            m = len(self.bins)
            if m <= 10:
                t = self.bins[-1] + 1
            elif m <= 12:
                t = self.bins[-1] + 2
            elif m <= 14:
                t = self.bins[-1] + 3
            elif m <= 16:
                t = self.bins[-1] + 4
            elif m <= 18:
                t = self.bins[-1] + 6
            elif m <= 22:
                t = self.bins[-1] + 8
            elif m <= 26:
                t = self.bins[-1] + 10
            elif m <= 30:
                t = self.bins[-1] + 15
            elif m <= 32:
                t = self.bins[-1] + 20
            elif m <= 34:
                t = self.bins[-1] + 25
            else:
                t = self.bins[-1] + 30
            self.bins.append(t)

    def _restrict_samples(self, restrictions):

        cond = np.zeros(len(self.metadata["obs"]["barcode"]), dtype=np.uint8)
        cond[self.cell_idx] = 1

        if restrictions is not None:
            for k, v in restrictions.items():

                if isinstance(v, list):
                    cond *= np.sum(np.stack([np.array(self.metadata["obs"][k]) == v1 for v1 in v]), axis=0).astype(
                        np.uint8)
                    print(k, v, np.sum(cond))
                else:
                    cond *= (np.array(self.metadata["obs"][k]) == v)
                    print(k, v, np.sum(cond))

        self.cell_idx = np.where(cond)[0]
        self.n_samples = len(self.cell_idx)

        for k in self.metadata["obs"].keys():
            self.metadata["obs"][k] = np.array(self.metadata["obs"][k])[self.cell_idx]

        print(f"Restricting samples; number of samples: {self.n_samples}")
        print(f"Number of subjects: {len(np.unique(self.metadata['obs']['SubID']))}")


    def _get_gene_index(self):

        N = len(self.metadata["var"]['gene_name'])
        self.metadata["var"]['percent_cells'] = np.ones(N)
        cond = self.metadata["var"]['percent_cells'] >= 0.0

        if self.remove_sex_chrom:
            cond *= self.metadata["var"]['gene_chrom'] != "X"
            cond *= self.metadata["var"]['gene_chrom'] != "Y"

        if self.protein_coding_only:
            cond *= self.metadata["var"]['protein_coding']

        self.gene_idx = np.where(cond)[0]
        self.gene_names = np.array(self.metadata["var"]['gene_name'])[self.gene_idx]

        self.n_genes = len(self.gene_idx)
        print(f"Sub-sampling genes. Number of genes is now {self.n_genes}")


    def _create_gene_cell_prop_ids(self):
        """"Create the gene and class ids. Will start with the gene ids, and then concatenate
        the cell property ids if requested"""
        gene_ids = torch.arange(0, self.n_genes).repeat(self.batch_size, 1)
        if self.n_cell_properties > 0:
            if self.cell_prop_same_ids:
                # this will project all the class related latent info onto the same subspace, simplifying analysis
                cell_prop_ids = torch.zeros((self.batch_size, self.n_cell_properties), dtype=torch.int64)
            else:
                cell_prop_ids = torch.arange(0, self.n_cell_properties).repeat(self.batch_size, 1)
        else:
            cell_prop_ids = None

        return gene_ids, cell_prop_ids

    def _get_batch_prop_vals(self):

        if self.batch_properties is None:
            return None

        n_batch_properties = len(self.batch_properties.keys())
        self.batch_labels = np.zeros((self.n_samples, n_batch_properties), dtype=np.int64)

        for n0 in range(self.n_samples):
            for n1, (k, prop) in enumerate(self.batch_properties.items()):
                cell_val = self.metadata["obs"][k][n0]
                self.batch_labels[n0, n1] = self.batch_properties[k]["index_dict"][cell_val]

    def _get_cell_prop_vals(self, ignore_index = -100):
        """Extract the cell property value for ach entry in the batch"""
        if self.n_cell_properties == 0:
            return None

        self.labels = np.zeros((self.n_samples, self.n_cell_properties), dtype=np.float32)

        for n0 in range(self.n_samples):
            for n1, (k, cell_prop) in enumerate(self.cell_properties.items()):
                cell_val = self.metadata["obs"][k][n0]

                if not cell_prop["discrete"]:
                    # continuous value
                    if cell_val > self.max_cell_prop_val or cell_val < -self.max_cell_prop_val or np.isnan(cell_val):
                        self.labels[n0, n1] = ignore_index
                    else:
                        self.labels[n0, n1] = cell_val
                else:
                    self.labels[n0, n1] = self.cell_properties[k]["index_dict"][cell_val]


    def _get_cell_batch_prop_vals_batch(self, batch_idx: List[int]):

        labels = self.labels[batch_idx] if self.cell_properties is not None else None
        batch_labels = self.batch_labels[batch_idx] if self.batch_properties is not None else None

        return labels, batch_labels

    def _get_gene_vals_batch(self, batch_idx: List[int]):

        target_gene_vals = np.zeros((self.batch_size, self.n_genes), dtype=np.float32)
        input_gene_vals = np.zeros_like(target_gene_vals)
        include_gene_mask = np.ones((self.batch_size, self.n_genes), dtype=np.int8)
        depths = np.zeros((self.batch_size, 2), dtype=np.float32)

        for n, i in enumerate(batch_idx):

            j = i if self.cell_idx is None else self.cell_idx[i]

            raw_gene_vals = np.memmap(
                self.data_path, dtype='uint8', mode='r', shape=(self.n_genes_original,), offset=j * self.offset
            )[self.gene_idx].astype(np.float32)

            idx = raw_gene_vals >= self.exclude_gene_val
            include_gene_mask[n, idx] = 0
            raw_gene_vals[idx] = 0

            target_gene_vals[n, :] = self._normalize(raw_gene_vals)

            if self.RDA:
                raw_gene_vals, depths[n, :] = self._randomly_sample_depth(raw_gene_vals)
            else:
                depths[n, :] = np.sum(raw_gene_vals)

            if self.embedding_strategy == "binned":
                input_gene_vals[n, :] = np.digitize(raw_gene_vals, self.bins)
            else:
                input_gene_vals[n, :] = self._normalize(raw_gene_vals)

        if depths is not None:
            depths = self._log_normalize(depths)

        # return two copies since we'll modify gene_vals but keep gene_targets as is
        return input_gene_vals, target_gene_vals, include_gene_mask, depths

    def _random_scaling(self, gene_vals: np.ndarray) -> np.ndarray:

        if self.random_scaling and self.training:
            alpha = 0.8 + 0.4 * np.random.rand()
            return alpha * gene_vals
        else:
            return gene_vals

    def _normalize(self, gene_vals: np.ndarray):

        x = self._normalize_total(gene_vals)
        x = self._log_normalize(x)
        return x

    def _normalize_total(self, x: np.ndarray, target_sum: float = 10_000) -> np.ndarray:
        return x * target_sum / (1 + np.sum(x)) if self.normalize_total else x

    def _log_normalize(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(x) if self.log_normalize else x


    def _randomly_sample_depth(self, gene_vals):

        depths = np.zeros(2, np.float32)
        depths[0] = np.sum(gene_vals)
        gamma = 0.5 if (depths[0] >= 1000 and self.training) else 0.0
        new_gene_vals = copy.deepcopy(gene_vals)

        if np.random.rand() < gamma:
            beta = np.clip(np.random.beta(2, 2), 0.1, 1.0)
            idx = np.nonzero(gene_vals)[0]
            new_gene_vals[idx] = np.clip(
                np.random.binomial(gene_vals[idx].astype(np.int64), beta), 0, self.exclude_gene_val - 1,
            ).astype(np.float32)

        depths[1] = np.sum(new_gene_vals)

        return new_gene_vals, depths


    def _prepare_data(self, batch_idx):

        # get input and target data, returned as numpy arrays
        input_gene_vals, target_gene_vals, include_gene_mask, depths = self._get_gene_vals_batch(batch_idx)
        if self.n_cell_properties > 0:
            cell_prop_vals, batch_prop_vals = self._get_cell_batch_prop_vals_batch(batch_idx)
        else:
            cell_prop_vals, cell_prop_mask = None, None

        return input_gene_vals, target_gene_vals, include_gene_mask, depths, cell_prop_vals, batch_prop_vals


    def __getitem__(self, batch_idx: Union[int, List[int]]):

        if isinstance(batch_idx, int):
            batch_idx = [batch_idx]

        # batch_indices = [self.cell_idx[i] for i in batch_idx]

        if len(batch_idx) != self.batch_size:
            raise ValueError("Index length not equal to batch_size")

        if self.masked_gene_selection == "VAE":
            n_genes_batch = self.n_genes
            n_targets = self.n_genes
        elif self.masked_gene_selection == "BERT":
            n_genes_batch = self.n_input
            n_targets = n_genes_batch
        elif self.training and self.n_input is not None:
            n_genes_batch = np.random.choice(np.arange(self.n_input // 4, self.n_input))
            n_targets = self.n_genes if self.n_targets is None else self.n_targets
        else:
            n_genes_batch = self.n_input
            n_targets = self.n_genes if self.n_targets is None else self.n_targets

        (
            pre_input_gene_vals,
            pre_target_gene_vals,
            include_gene_mask,
            depths,
            cell_prop_vals,
            batch_prop_vals,
        ) = self._prepare_data(batch_idx)

        # select which genes to use as input, and which to mask
        # initialize gene ids ids at padding value

        gene_ids = self.n_genes * np.ones((self.batch_size, n_genes_batch), dtype=np.int64)
        gene_vals = np.zeros((self.batch_size, n_genes_batch), dtype=np.float32)
        gene_target_ids = np.zeros((self.batch_size, n_targets), dtype=np.int64)
        gene_target_vals = np.zeros((self.batch_size, n_targets), dtype=np.float32)
        target_mask = np.ones((self.batch_size, n_targets), dtype=np.float32) # 1 = include, 0 = exclude

        #n_input = n_genes_batch if not self.RDA else n_genes_batch # + 2  TODO: FIX THIS
        padding_mask = np.zeros((self.batch_size, n_genes_batch), dtype=np.float32)

        for n in range(self.batch_size):

            possible_input_genes = np.where(include_gene_mask[n, :])[0]

            if self.masked_gene_selection == "BERT":
                p_mask = 0.20
                N = np.minimum(len(possible_input_genes), n_genes_batch)
                input_idx = np.random.choice(possible_input_genes, N, replace=False)
                gene_ids[n, :N] = gene_target_ids[n, :N] = input_idx
                gene_vals[n, :N] = gene_target_vals[n, :N] = pre_input_gene_vals[n, input_idx]

                masked_out_genes = np.random.choice(n_genes_batch, int(N * p_mask), replace=False)
                padding_mask[n, masked_out_genes] = 1

                if N < n_genes_batch:
                    gene_ids[n, N:] = gene_target_ids[n, N:] = self.n_genes
                    gene_vals[n, N:] = gene_target_vals[n, N:] = 0
                    padding_mask[n, N:] = 1
                    target_mask[n, N:] = 0.0

            elif self.masked_gene_selection == "VAE":

                # gene_ids[n, :] = gene_target_ids[n, :] = np.arange(n_genes_batch) # not really needed
                gene_vals[n, :] = gene_target_vals[n, :] = pre_input_gene_vals[n, :]
                target_mask[n, :] = include_gene_mask[n, :]

            else:
                padding_mask = None
                replace = False if n_targets <= len(possible_input_genes) else True
                target_idx = np.random.choice(possible_input_genes, n_targets, replace=replace)

                gene_target_vals[n, :] = pre_target_gene_vals[n, target_idx]
                gene_target_ids[n, :] = target_idx

                if self.masked_gene_selection == "exclusive":
                    remainder_idx = list(set(possible_input_genes) - set(target_idx))
                    #remainder_idx = list(set(possible_input_genes) - set(target_idx[50:]))
                    #if n_genes_batch > len(remainder_idx):
                    #    remainder_idx = possible_input_genes

                elif self.masked_gene_selection == "non_exclusive":
                    remainder_idx = possible_input_genes

                if n_genes_batch <= len(remainder_idx):
                    input_idx = np.random.choice(remainder_idx, n_genes_batch, replace=False)
                    gene_ids[n, :] = input_idx
                    gene_vals[n, :] = pre_input_gene_vals[n, input_idx]
                else:
                    n_remain = len(remainder_idx)
                    gene_ids[n, :n_remain] = remainder_idx
                    gene_vals[n, :n_remain] = pre_input_gene_vals[n, remainder_idx]

                    gene_ids[n, n_remain:] = self.n_genes
                    gene_vals[n, n_remain:] = 0.0

        if self.masked_gene_selection in ["exclusive", "non_exclusive"]:
            padding_mask = None
            target_mask = None
        elif self.masked_gene_selection in ["VAE"]:
            padding_mask = None
            gene_ids = None
            gene_target_ids = None

        batch = (
            gene_ids,
            gene_target_ids,
            gene_vals,
            gene_target_vals,
            padding_mask,
            target_mask,
            depths,
            cell_prop_vals,
            batch_prop_vals,
            batch_idx,
        )

        return batch



class DataModule(pl.LightningDataModule):

    # data_path: Path to directory with preprocessed data.
    # classify: Name of column from `obs` table to add classification task with. (optional)
    # Fraction of median genes to mask for prediction.
    # batch_size: Dataloader batch size
    # num_workers: Number of workers for DataLoader.

    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        train_idx: List[int],
        test_idx: List[int],
        batch_size: int = 32,
        num_workers: int = 16,
        n_targets: Optional[int] = 100,
        n_input: int = 100,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        cell_prop_same_ids: bool = False,
        remove_sex_chrom: bool = False,
        protein_coding_only: bool = False,
        n_bins: Optional[int] = False,
        embedding_strategy: Literal["binned", "continuous", "film"] = "continuous",
        RDA: bool = False,
        cell_restrictions: Optional[Dict[str, Any]] = None,
        masked_gene_selection: Literal["BERT", "exclusive", "non_exclusive"] = "BERT",
    ):
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_targets = n_targets
        self.n_input = n_input
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.cell_prop_same_ids = cell_prop_same_ids
        self.remove_sex_chrom = remove_sex_chrom
        self.protein_coding_only = protein_coding_only
        self.embedding_strategy = embedding_strategy
        self.n_bins = n_bins
        self.RDA = RDA
        self.cell_restrictions = cell_restrictions
        self.masked_gene_selection = masked_gene_selection

        #self._get_cell_prop_info()
        self._get_cell_prop_batch_indices()


    def _get_cell_prop_batch_indices(self):

        metadata = pickle.load(open(self.metadata_path, "rb"))
        for properties in [self.cell_properties, self.batch_properties]:

            if properties is not None:
                for k, cell_prop in properties.items():
                    if cell_prop["discrete"] and (cell_prop["values"] is None or cell_prop["values"] == -1):
                        cell_vals = metadata["obs"][k]
                        unique_list = np.unique(cell_vals)
                        # remove key terms for not available, nan, etc..
                        unique_list = set(unique_list).difference(set(["na", "NA", "nan", "NAN"]))
                        unique_list = list(unique_list)
                        cell_prop["index_dict"] = {"na": -100, "NA": -100, "nan": -100, "NAN": -100}
                        for n, v in enumerate(unique_list):
                            cell_prop["index_dict"][v] = n
                        cell_prop["n_values"] = len(unique_list)

        del metadata


    def _get_cell_prop_info(self, max_cell_prop_val = 999):
        """Extract the list of uniques values for each cell property (e.g. sex, cell type, etc.) to be predicted"""

        self.n_cell_properties = len(self.cell_properties) if self.cell_properties is not None else 0

        metadata = pickle.load(open(self.metadata_path, "rb"))

        # not a great place for this, but needed
        self.n_genes = len(metadata["var"]["gene_name"])

        if self.n_cell_properties > 0:

            for k, cell_prop in self.cell_properties.items():
                # skip if required field are already present as this function can be called multiple
                # times if using multiple GPUs

                cell_vals = metadata["obs"][k]

                if "freq" in self.cell_properties[k] or "mean" in self.cell_properties[k]:
                    continue
                if not cell_prop["discrete"]:
                    # for cell properties with continuous value, determine the mean/std for normalization
                    # remove nans, negative values, or anything else suspicious
                    idx = [n for n, cv in enumerate(cell_vals) if cv >= 0 and cv < max_cell_prop_val]
                    self.cell_properties[k]["mean"] = np.mean(cell_vals[idx])
                    self.cell_properties[k]["std"] = np.std(cell_vals[idx])

                elif cell_prop["discrete"] and cell_prop["values"] is None:
                    # for cell properties with discrete value, determine the possible values if none were supplied
                    # and find their distribution
                    unique_list, counts = np.unique(cell_vals, return_counts=True)
                    # remove nans, negative values, or anything else suspicious
                    idx = [
                        n for n, u in enumerate(unique_list) if (
                            isinstance(u, str) or (u >= 0 and u < max_cell_prop_val)
                        )
                    ]
                    self.cell_properties[k]["values"] = unique_list[idx]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])
                    print("CELL PROP INFO",k, self.cell_properties[k]["freq"])

                elif cell_prop["discrete"] and cell_prop["values"] is not None:

                    counts = []
                    for p in cell_prop["values"]:
                        print(p, np.sum(np.array(cell_vals) == p))
                        counts.append(np.sum(np.array(cell_vals) == p))
                    counts = np.array(counts)
                    self.cell_properties[k]["freq"] = counts / np.mean(counts)
                    print("CELL PROP INFO", k, self.cell_properties[k]["freq"])

        else:
            self.cell_prop_dist = None


    def setup(self, stage):

        self.train_dataset = SingleCellDataset(
            self.data_path,
            self.metadata_path,
            self.train_idx,
            cell_properties=self.cell_properties,
            batch_properties=self.batch_properties,
            n_targets=self.n_targets,
            n_input=self.n_input,
            batch_size=self.batch_size,
            cell_prop_same_ids=self.cell_prop_same_ids,
            embedding_strategy = self.embedding_strategy,
            n_bins=self.n_bins,
            protein_coding_only=self.protein_coding_only,
            remove_sex_chrom=self.remove_sex_chrom,
            cell_restrictions=self.cell_restrictions,
            masked_gene_selection=self.masked_gene_selection,
            RDA=self.RDA,
            training=True,
            n_genes_per_input=4_000,
        )
        self.val_dataset = SingleCellDataset(
            self.data_path,
            self.metadata_path,
            self.test_idx,
            cell_properties=self.cell_properties,
            batch_properties=self.batch_properties,
            n_targets=self.n_targets,
            n_input=self.n_input,
            batch_size=self.batch_size,
            cell_prop_same_ids=self.cell_prop_same_ids,
            embedding_strategy=self.embedding_strategy,
            n_bins=self.n_bins,
            protein_coding_only=self.protein_coding_only,
            remove_sex_chrom=self.remove_sex_chrom,
            cell_restrictions=self.cell_restrictions,
            masked_gene_selection=self.masked_gene_selection,
            RDA=self.RDA,
            training=False,
            n_genes_per_input=4_000,
        )

        self.n_genes = self.train_dataset.n_genes
        print(f"number of genes {self.n_genes}")


    # return the dataloader for each split
    def train_dataloader(self):
        sampler = BatchSampler(
            RandomSampler(self.train_dataset),
            batch_size=self.train_dataset.batch_size,
            drop_last=True,
        )
        dl = DataLoader(
            self.train_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=self.num_workers,
        )
        return dl

    def val_dataloader(self):
        sampler = BatchSampler(
            RandomSampler(self.val_dataset),
            batch_size=self.val_dataset.batch_size,
            drop_last=True,
        )
        dl = DataLoader(
            self.val_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=self.num_workers,
        )
        return dl

