from typing import Any, Dict, List, Optional, Tuple, Union

import os
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
import time

class SingleCellDataset(Dataset):
    def __init__(
        self,
        adata_path: str,
        cell_idx: np.array,
        gene_idx: Any,
        cell_properties: Optional[Dict[str, Any]] = None,
        n_mask: int = 100,
        batch_size: int = 32,
        normalize_total: Optional[float] = 10_000,
        log_normalize: bool = True,
        bin_gene_count: bool = True,
        n_gene_bins: int = 16,
        pin_memory: bool = False,
        cell_prop_same_ids: bool = False,
        max_cell_prop_val: float = 999,
        cutmix_pct: float = 0.0,
        mixup: bool = False,
        n_genes_per_input: int = 4_000,
    ):

        self.adata = sc.read_h5ad(adata_path, backed="r")
        self.cell_idx = cell_idx
        self.gene_idx = gene_idx
        self.n_samples = len(cell_idx)
        self.n_genes = np.sum(gene_idx)

        self.cell_properties = cell_properties
        self.n_cell_properties = len(cell_properties) if cell_properties is not None else 0

        self.n_mask = n_mask
        self.batch_size = batch_size

        self.normalize_total = normalize_total
        self.log_normalize = log_normalize
        self.bin_gene_count = bin_gene_count
        self.n_gene_bins = n_gene_bins
        self.pin_memory = pin_memory
        self.cell_prop_same_ids = cell_prop_same_ids
        self.max_cell_prop_val = max_cell_prop_val
        self.cutmix_pct = cutmix_pct
        self.mixup = mixup
        self.n_genes_per_input = n_genes_per_input

        # possibly use for embedding the gene inputs
        self.cell_classes = np.array(['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo'])

        self._create_data()

        self.bins = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 13, 16, 22, 35, 60, 9999]


    def __len__(self):
        return self.n_samples


    def _get_cell_prop_vals(self, adata: Any):
        """Extract the cell property value for ach entry in the batch"""
        if self.n_cell_properties == 0:
            return None

        cell_prop_vals = np.zeros((adata.shape[0], self.n_cell_properties), dtype=np.float32)
        cell_class_id = np.zeros((adata.shape[0]), dtype=np.uint8)

        for n0 in range(adata.shape[0]):
            for n1, (k, cell_prop) in enumerate(self.cell_properties.items()):
                cell_val = adata.obs[k][n0]
                if not cell_prop["discrete"]:
                    # continuous value
                    if np.abs(cell_val) > self.max_cell_prop_val:
                        cell_prop_vals[n0, n1] = -9999
                    else:
                        # normalize
                        cell_prop_vals[n0, n1] = (cell_val - cell_prop["mean"]) / cell_prop["std"]
                else:
                    # discrete value
                    idx = np.where(cell_val == np.array(cell_prop["values"]))[0]
                    # cell property values of -1 will imply N/A, and will be masked out
                    if len(idx) == 0:
                        cell_prop_vals[n0, n1] = -9999
                    else:
                        cell_prop_vals[n0, n1] = idx[0]

            idx = np.where(adata.obs["class"][n0] == self.cell_classes)[0]
            cell_class_id[n0] = idx[0]

        return cell_prop_vals, cell_class_id

    def _create_data(self, max_cells: Optional[int] = 40_000, chunk_size: int = 2_000):

        self.gene_counts = np.zeros((0, self.n_genes), dtype=np.uint8)
        self.labels = np.zeros((0, self.n_cell_properties), dtype=np.float32)
        self.cell_class = np.zeros((0,), dtype=np.uint8)

        n_cells = len(self.cell_idx) if max_cells is None else np.minimum(len(self.cell_idx), max_cells)
        n = n_cells // chunk_size + 1 if n_cells % chunk_size > 0 else n_cells // chunk_size

        for i in range(n):
            t0 = time.time()

            m = np.minimum((i + 1) * chunk_size, n_cells)
            current_idx = list(range(i * chunk_size, m))
            include_idx = [i for i, j in enumerate(current_idx) if j in self.cell_idx]
            if len(include_idx) == 0:
                continue

            print(f"Creating data, chunk {i}")
            temp = self.adata[i * chunk_size: m, self.gene_idx].to_memory(copy=True)
            temp = temp[include_idx]
            counts = np.array(temp.X.todense())
            print(f"Time taken {time.time() - t0:2.2f}")
            counts[counts >= 255] = 255
            counts = counts.astype(np.uint8)

            self.gene_counts = np.concatenate((self.gene_counts, counts), axis=0)
            cell_prop_vals, cell_class_ids = self._get_cell_prop_vals(temp)
            self.labels = np.concatenate((self.labels, cell_prop_vals), axis=0)
            self.cell_class = np.concatenate((self.cell_class, cell_class_ids), axis=-1)


        self.n_samples = self.gene_counts.shape[0]

        print(f"Data created. Number of cells: {self.gene_counts.shape[0]}")


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

    def _prepare_data(self, batch_idx):

        # get input and target data, returned as numpy arrays
        input_gene_vals, target_gene_vals = self._get_gene_vals_batch(batch_idx)
        cell_prop_vals, cell_class_id = self._get_cell_prop_vals_batch(batch_idx)

        # If specified, perform data augmentation mixup or cutmix
        if self.mixup:
            gene_vals, gene_targets, cell_prop_vals = self._mixup(gene_vals, gene_targets, cell_prop_vals)
        elif self.cutmix_pct > 0:
            gene_vals, gene_targets, cell_prop_vals = self._cutmix(gene_vals, gene_targets, cell_prop_vals)

        """
        zero_idx = np.where(gene_vals == 0)
        key_padding_mask = torch.zeros_like(gene_vals).detach()
        key_padding_mask[zero_idx[0], zero_idx[1]] = 1.0
        """

        return input_gene_vals, target_gene_vals, cell_prop_vals, cell_class_id
        #return gene_vals, key_padding_mask, gene_targets, cell_prop_vals, cell_class_id, zero_idx

    def _get_cell_prop_vals_batch(self, batch_idx: List[int]):

        return self.labels[batch_idx], self.cell_class[batch_idx]

    def _get_gene_vals_batch(self, batch_idx: List[int]):

        target_gene_vals = np.zeros((self.batch_size, self.n_genes), dtype=np.float32)
        input_gene_vals = np.zeros_like(target_gene_vals)

        for n, i in enumerate(batch_idx):
            if self.bin_gene_count:
                input_gene_vals[n, :] = self._bin_gene_count(self.gene_counts[i, :])
                target_gene_vals[n, :] = self._normalize(self.gene_counts[i, :])
            elif self.normalize_total or self.log_normalize:
                input_gene_vals[n, :] = self._normalize(self.gene_counts[i, :])
                target_gene_vals[n, :] = input_gene_vals[n, :]


        # return two copies since we'll modify gene_vals but keep gene_targets as is
        return input_gene_vals, target_gene_vals

    def _bin_gene_count(self, x: np.ndarray) -> np.ndarray:
        return np.digitize(x.astype(np.float32), self.bins)


    def _normalize(self, x: np.ndarray) -> np.ndarray:

        x = x.astype(np.float32)
        x = x * self.normalize_total / np.sum(x) if self.normalize_total is not None else x
        x = np.log1p(x) if self.log_normalize else x
        return x

    def _mixup(self, gene_vals, gene_targets, cell_prop_vals):

        # determine the cells with no missing values
        p = cell_prop_vals[:, :, 0].detach().numpy()
        good_idx = list(np.where(np.sum(p < -999, axis=1) == 0)[0])
        good_set = set(good_idx)

        new_cell_prop_vals = torch.zeros_like(cell_prop_vals)
        new_gene_targets = torch.zeros_like(gene_targets)
        new_gene_vals = torch.zeros_like(gene_vals)

        for n in range(self.batch_size):
            if n in good_idx:
                # randomly choose partner
                j = np.random.choice(list(good_set.difference(set([n]))))
                # set mix percentage
                alpha = np.clip(np.random.exponential(0.05), 0.0, 0.25)
                new_gene_vals[n, :] = (1 - alpha) * gene_vals[n, :] + alpha * gene_vals[j, :]

                # take the weighted average of the targets
                new_cell_prop_vals[n, :] = (1 - alpha) * cell_prop_vals[n, :] + alpha * cell_prop_vals[j, :]
                new_gene_targets[n, :] = (1 - alpha) * gene_targets[n, :] + alpha * gene_targets[j, :]

            else:
                new_cell_prop_vals[n, :] = cell_prop_vals[n, :]
                new_gene_targets[n, :] = gene_targets[n, :]
                new_gene_vals[n, :] = gene_vals[n, :]

        return new_gene_vals, new_gene_targets, new_cell_prop_vals

    def _cutmix(self, gene_vals, gene_targets, cell_prop_vals, continuous_block: bool = False):

        # determine the cells with no missing values
        p = cell_prop_vals[:, :, 0].detach().numpy()
        good_idx = list(np.where(np.sum(p < -999, axis=1) == 0)[0])
        good_set = set(good_idx)

        new_cell_prop_vals = torch.zeros_like(cell_prop_vals)
        new_gene_targets = torch.zeros_like(gene_targets)
        new_gene_vals = torch.zeros_like(gene_vals)

        for n in range(self.batch_size):
            if n in good_idx and np.random.rand() < self.cutmix_pct:
                # randomly choose partner
                j = np.random.choice(list(good_set.difference(set([n]))))
                # set mix percentage
                alpha = np.random.uniform(0.005, 0.995)

                # mix-up gene values
                new_gene_vals[n, :] = gene_vals[n, :]
                if continuous_block:
                    start_idx = np.random.randint(0, int(alpha * self.n_genes) - 1)
                    end_idx = start_idx + int((1 - alpha) * self.n_genes)
                    new_gene_vals[n, :] = gene_vals[n, :]
                    new_gene_vals[n, start_idx : end_idx] = gene_vals[j, start_idx: end_idx]
                else:
                    idx_mix = np.random.choice(self.n_genes, int(alpha * self.n_genes), replace=False)
                    new_gene_vals[n, idx_mix] = gene_vals[j, idx_mix]

                # ensure it has enough non-zero entries if not; then revert
                if torch.sum(new_gene_vals[n, :] > 0) < self.n_mask:
                    new_gene_vals[n, :] = gene_vals[n, :]
                    continue

                # take the weighted average of the targets
                new_cell_prop_vals[n, :] = (1 - alpha) * cell_prop_vals[n, :] + alpha * cell_prop_vals[j, :]
                new_gene_targets[n, :] = (1 - alpha) * gene_targets[n, :] + alpha * gene_targets[j, :]


            else:
                new_cell_prop_vals[n, :] = cell_prop_vals[n, :]
                new_gene_targets[n, :] = gene_targets[n, :]
                new_gene_vals[n, :] = gene_vals[n, :]

        return new_gene_vals, new_gene_targets, new_cell_prop_vals

    def __getitem__(self, idx: Union[int, List[int]]):

        if isinstance(idx, int):
            idx = [idx]

        if len(idx) != self.batch_size:
            raise ValueError("Index length not equal to batch_size")

        """
        gene_vals, key_padding_mask, gene_targets, cell_prop_vals, cell_class_id, zero_idx = self._prepare_data(idx)
        """
        pre_input_gene_vals, pre_target_gene_vals, cell_prop_vals, cell_class_id = self._prepare_data(idx)

        # select which genes to use as input, and which to mask
        # initialize gene ids ids at padding value
        gene_ids = self.n_genes * np.ones((self.batch_size, self.n_genes_per_input, ), dtype = np.int64)
        padding_mask = np.ones((self.batch_size, self.n_genes_per_input,), dtype=np.float32)
        gene_vals = np.zeros((self.batch_size, self.n_genes_per_input), dtype=np.float32)
        gene_target_ids = np.zeros((self.batch_size, self.n_mask), dtype=np.int64)
        gene_target_vals = np.zeros((self.batch_size, self.n_mask), dtype=np.float32)

        for n in range(self.batch_size):
            nonzero_idx = np.nonzero(pre_input_gene_vals[n, :])[0]
            mask_idx = np.random.choice(nonzero_idx, self.n_mask, replace=False)
            gene_target_vals[n, :] = pre_target_gene_vals[n, mask_idx]
            gene_target_ids[n, :] = mask_idx
            remaineder_idx = list(set(nonzero_idx) - set(mask_idx))
            if len(remaineder_idx) <= self.n_genes_per_input:
                gene_ids[n, :len(remaineder_idx)] = remaineder_idx
                gene_vals[n, :len(remaineder_idx)] = pre_input_gene_vals[n, remaineder_idx]
                padding_mask[n, :len(remaineder_idx)] = 0.0
            else:
                idx = np.random.choice(remaineder_idx, self.n_genes_per_input, replace=False)
                gene_ids[n, :] = idx
                gene_vals[n, :] = pre_input_gene_vals[n, idx]
                padding_mask[n, :] = 0.0

        # how to query the latent output in order to predict cell properties
        if self.cell_prop_same_ids:
            # this will project all the class related latent info onto the same subspace, simplifying analysis
            cell_prop_ids = np.zeros((self.batch_size, self.n_cell_properties), dtype=np.int64)
        else:
            cell_prop_ids = np.tile(np.arange(0, self.n_cell_properties)[None, :], (self.batch_size, 1))

        """
        batch = (
            torch.from_numpy(gene_ids),
            torch.from_numpy(gene_target_ids),
            torch.from_numpy(cell_prop_ids),
            torch.from_numpy(gene_vals),
            torch.from_numpy(gene_target_vals),
            torch.from_numpy(padding_mask),
            torch.from_numpy(cell_prop_vals),
            torch.from_numpy(cell_class_id),
        )
        """
        batch = (
            gene_ids,
            gene_target_ids,
            cell_prop_ids,
            gene_vals,
            gene_target_vals,
            padding_mask,
            cell_prop_vals,
            cell_class_id,
        )


        if self.pin_memory:
            for tensor in batch:
                tensor.pin_memory()

        return batch


class DataModule(pl.LightningDataModule):

    # data_path: Path to directory with preprocessed data.
    # classify: Name of column from `obs` table to add classification task with. (optional)
    # Fraction of median genes to mask for prediction.
    # batch_size: Dataloader batch size
    # num_workers: Number of workers for DataLoader.

    def __init__(
        self,
        adata_path: str,
        batch_size: int = 32,
        num_workers: int = 1,
        n_mask: int = 100,
        cell_properties: Optional[Dict[str, Any]] = None,
        cell_prop_same_ids: bool = False,
        cutmix_pct: float = 0.0,
        mixup: bool = False,
        bin_gene_count: bool = False,
        rank_order: bool = False,
        split_train_test_by_subject: bool = True,
        train_pct: float = 0.90,
        protein_coding_only: bool = True,
        min_percent_of_cells: float = 2.0,
    ):
        super().__init__()
        self.adata_path = adata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_mask = n_mask
        self.rank_order = rank_order
        self.cell_properties = cell_properties
        self.cell_prop_same_ids = cell_prop_same_ids
        self.cutmix_pct = cutmix_pct
        self.mixup = mixup
        self.bin_gene_count = bin_gene_count
        self.split_train_test_by_subject = split_train_test_by_subject
        self.train_pct = train_pct
        self.protein_coding_only = protein_coding_only
        self.min_percent_of_cells = min_percent_of_cells

        self.adata = sc.read_h5ad(adata_path, backed="r")

        self._train_test_splits()
        self._get_gene_index()

        self._get_cell_prop_info()

    def _get_gene_index(self):

        self.gene_idx = self.adata.var["percent_cells"] > self.min_percent_of_cells
        if self.protein_coding_only:
            self.gene_idx *= self.adata.var["protein_coding"]

        self.n_genes = np.sum(self.gene_idx)
        print(f"Number of genes: {self.n_genes}")
    def _get_cell_prop_info(self, max_cell_prop_val = 999):
        """Extract the list of uniques values for each cell property (e.g. sex, cell type, etc.) to be predicted"""

        self.n_cell_properties = len(self.cell_properties) if self.cell_properties is not None else 0

        if self.n_cell_properties > 0:

            for k, cell_prop in self.cell_properties.items():
                # skip if required field are already present as this function can be called multiple
                # times if using multiple GPUs
                if "freq" in self.cell_properties[k] or "mean" in self.cell_properties[k]:
                    continue
                if not cell_prop["discrete"]:
                    # for cell properties with continuous value, determine the mean/std for normalization
                    cell_vals = self.adata.obs[k]
                    # remove nans, negative values, or anything else suspicious
                    idx = [n for n, cv in enumerate(cell_vals) if cv >= 0 and cv < max_cell_prop_val]
                    self.cell_properties[k]["mean"] = np.mean(cell_vals[idx])
                    self.cell_properties[k]["std"] = np.std(cell_vals[idx])

                elif cell_prop["discrete"] and cell_prop["values"] is None:
                    # for cell properties with discrete value, determine the possible values if none were supplied
                    # and find their distribution
                    unique_list, counts = np.unique(self.adata.obs[k], return_counts=True)
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
                    unique_list, counts = np.unique(self.adata.obs[k], return_counts=True)
                    idx = [n for n, u in enumerate(unique_list) if u in cell_prop["values"]]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])
                    print("CELL PROP INFO", k, self.cell_properties[k]["freq"])

        else:
            self.cell_prop_dist = None



    def _train_test_splits(self):

        if self.split_train_test_by_subject:
            sub_ids = np.unique(self.adata.obs["SubID"].values)
            np.random.shuffle(sub_ids)
            n = len(sub_ids)
            train_ids = sub_ids[: int(n * self.train_pct)]
            test_ids = sub_ids[int(n * self.train_pct):]
            self.train_idx = [n for n, s_id in enumerate(self.adata.obs["SubID"].values) if s_id in train_ids]
            self.test_idx = [n for n, s_id in enumerate(self.adata.obs["SubID"].values) if s_id in test_ids]
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
        #self.train_idx = np.sort(self.train_idx)
        #self.test_idx = np.sort(self.test_idx)


    def setup(self, stage):

        self.train_dataset = SingleCellDataset(
            self.adata_path,
            self.train_idx,
            self.gene_idx,
            cell_properties=self.cell_properties,
            n_mask=self.n_mask,
            batch_size=self.batch_size,
            pin_memory=False,
            cell_prop_same_ids=self.cell_prop_same_ids,
            cutmix_pct=self.cutmix_pct,
            mixup=self.mixup,
            bin_gene_count=self.bin_gene_count,
        )

        self.val_dataset = SingleCellDataset(
            self.adata_path,
            self.test_idx,
            self.gene_idx,
            cell_properties=self.cell_properties,
            n_mask=self.n_mask,
            batch_size=self.batch_size,
            pin_memory=False,
            cell_prop_same_ids=self.cell_prop_same_ids,
            cutmix_pct=0.0,
            mixup=False,
            bin_gene_count=self.bin_gene_count,
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
            pin_memory=False,
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
            pin_memory=False,
        )
        return dl

