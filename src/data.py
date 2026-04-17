from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import copy
import pickle
import torch
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
        batch_size: int = 64,
        max_cell_prop_val: float = 999,
        remove_sex_chrom: bool = False,
        protein_coding_only: bool = False,
        cell_restrictions: Optional[Dict[str, Any]] = None,
        top_k_genes: Optional[int] = None,
        gene_idx: Optional[List[int]] = None,
        group_balancing: Optional[str] = None,
        training: bool = True,
    ):

        self.metadata = pickle.load(open(metadata_path, "rb"))

        self.data_path = data_path
        self.cell_idx = cell_idx if cell_idx is not None else np.arange(len(self.metadata["obs"]["SubID"]))
        self.n_samples = len(self.cell_idx)
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.batch_size = batch_size
        self.group_balancing = group_balancing

        self.training = training

        self._restrict_samples(cell_restrictions)

        if self.group_balancing is not None:
            self._define_groups()

        print(f"Number of cells {self.n_samples}")
        if "gene_name" in self.metadata["var"].keys():
            self.n_genes_original = len(self.metadata["var"]["gene_name"])
        else:
            self.n_genes_original = len(self.metadata["var"])

        self.n_cell_properties = len(cell_properties) if cell_properties is not None else 0
        self.batch_size = batch_size

        self.max_cell_prop_val = max_cell_prop_val
        self.remove_sex_chrom = remove_sex_chrom
        self.protein_coding_only = protein_coding_only
        self.top_k_genes = top_k_genes

        # offset is needed for memmap loading
        self.offset = 1 * self.n_genes_original  # UINT8 is 1 bytes

        # this will down-sample the number if genes if specified
        if gene_idx is None:
            self._gene_stats()
            self._get_gene_index()
        else:
            self.gene_idx = gene_idx
            self.n_genes = len(self.gene_idx )
        self._get_cell_prop_vals()
        self._get_batch_prop_vals()

        # can remove
        self.metadata = None

    def __len__(self):
        return self.n_samples


    def _define_groups(self):
        """used to address clsss/attribute imabalance
        Simple data balancing achieves competitive worst-group-accuracy
        https://arxiv.org/pdf/2110.14503.pdf"""

        if self.group_balancing == "cd":
            self.group_idx = {}
            for c in range(1, 5):
                for d in range(2):
                    cond = (self.metadata["obs"]["Dementia"] == d) * (self.metadata["obs"]["CERAD"] == c)
                    idx = np.where(cond)[0]
                    i = (c-1) * 10 + d
                    self.group_idx[i] = idx.tolist()

            cond = (self.metadata["obs"]["Dementia"] < 0) + (self.metadata["obs"]["CERAD"] < 0)
            idx = np.where(cond)[0]
            self.group_idx[999] = idx.tolist()

        elif self.group_balancing == "bd":
            self.group_idx = {}
            for b in range(7):
                for d in range(2):
                    cond = (self.metadata["obs"]["Dementia"] == d) * (
                                self.metadata["obs"]["BRAAK_AD"] == b)
                    idx = np.where(cond)[0]
                    i = b * 10 + d
                    self.group_idx[i] = idx.tolist()

            cond = (self.metadata["obs"]["Dementia"] < 0) + (self.metadata["obs"]["BRAAK_AD"] < 0)
            idx = np.where(cond)[0]
            #print("GROUP 999 EXCLUDED")
            self.group_idx[999] = idx.tolist()

        elif self.group_balancing == "bd_apdaptive":
            self.group_idx = {}
            for b in range(7):
                for d in range(3):
                    cond = (self.metadata["obs"]["Dementia_graded"] == d * 0.5) * (self.metadata["obs"]["BRAAK_AD"] == b)
                    idx = np.where(cond)[0]
                    i = b * 10 + d
                    self.group_idx[i] = idx.tolist()
                    if d == 0:
                        self.group_idx[i+0.1] = idx.tolist()
                    elif d == 2:
                        for k in range(5):
                            self.group_idx[i + k*0.1] = idx.tolist()


            cond = (self.metadata["obs"]["Dementia"] < 0) + (self.metadata["obs"]["BRAAK_AD"] < 0)
            idx = np.where(cond)[0]
            #print("GROUP 999 EXCLUDED")
            self.group_idx[999] = idx.tolist()


        elif self.group_balancing == "bcd":
            self.group_idx = {}
            for b in range(7):
                for c in range(5):
                    for d in range(2):
                        cond = (self.metadata["obs"]["Dementia"] == d) * (
                                    self.metadata["obs"]["BRAAK_AD"] == b) * (
                                    self.metadata["obs"]["CERAD"] == c)
                        idx = np.where(cond)[0]
                        i = c * 100 + b * 10 + d
                        if len(idx) >= 100:
                            self.group_idx[i] = idx.tolist()

            cond = self.metadata["obs"]["Dementia"] < -99
            idx = np.where(cond)[0]
            self.group_idx[1000] = idx.tolist()
            cond = self.metadata["obs"]["BRAAK_AD"] < -99
            idx = np.where(cond)[0]
            self.group_idx[1001] = idx.tolist()
            cond = self.metadata["obs"]["CERAD"] < -99
            idx = np.where(cond)[0]
            self.group_idx[1002] = idx.tolist()

        elif self.group_balancing == "bc":
            self.group_idx = {}
            for b in range(7):
                for c in range(5):
                    cond = (self.metadata["obs"]["BRAAK_AD"] == b) * (
                            self.metadata["obs"]["CERAD"] == c)
                    idx = np.where(cond)[0]
                    i = c * 10 + b
                    if len(idx) >= 10:
                        self.group_idx[i] = idx.tolist()

            cond = self.metadata["obs"]["BRAAK_AD"] < -99
            idx = np.where(cond)[0]
            self.group_idx[101] = idx.tolist()
            cond = self.metadata["obs"]["CERAD"] < -99
            idx = np.where(cond)[0]
            self.group_idx[102] = idx.tolist()


        elif self.group_balancing == "pd":
            self.group_idx = {}
            for b in range(7):
                cond = (self.metadata["obs"]["path_braak_lb"] == b)
                idx = np.where(cond)[0]
                i = b
                self.group_idx[i] = idx.tolist()

        elif self.group_balancing == "pd_condensed":
            self.group_idx = {}
            for b in range(4):
                cond = (self.metadata["obs"]["path_braak_lb_condensed"] == b)
                idx = np.where(cond)[0]
                if len(idx) > 0:
                    i = b
                    self.group_idx[i] = idx.tolist()

        elif self.group_balancing == "pd_condensed_v2":
            self.group_idx = {}
            for b in range(3):
                cond = (self.metadata["obs"]["path_braak_lb_condensed_v2"] == b)
                idx = np.where(cond)[0]
                if len(idx) > 0:
                    i = b
                    self.group_idx[i] = idx.tolist()

        elif self.group_balancing == "pd_condensed_brain_region":
            brain_regions = ['DMNX', 'GPI', 'PFC', 'PMC', 'PVC']
            self.group_idx = {}
            for b in range(6):
                for r, br in enumerate(brain_regions):
                    cond = (self.metadata["obs"]["path_braak_lb_condensed"] == b) * (
                                self.metadata["obs"]["brain_reg_w_gt"] == br)
                    idx = np.where(cond)[0]
                    i = b * 10 + r
                    self.group_idx[i] = idx.tolist()

        elif self.group_balancing == "brain_region":
            brain_regions = ['DMNX', 'GPI', 'PFC', 'PMC', 'PVC']
            self.group_idx = {}
            for r, br in enumerate(brain_regions):
                cond = self.metadata["obs"]["brain_reg_w_gt"] == br
                idx = np.where(cond)[0]
                self.group_idx[r] = idx.tolist()

        # only include groups with a sufficient (e.g. 100) number of cells
        self.group_idx = {k: v for k, v in self.group_idx.items() if len(v) >= 5}

        print("Size of each group...")
        for k, v in self.group_idx.items():
            print(f"{k}: {len(v)}")

        n_groups = len(self.group_idx)
        self.cell_per_group = self.batch_size // n_groups

    def _gene_stats(self):

        print("Calculating gene stats...")
        N = 100_000 # maximum number of cells to sample
        counts = np.zeros(self.n_genes_original, dtype=np.float32)
        sums = np.zeros(self.n_genes_original, dtype=np.float32)
        idx = self.cell_idx[:N] if len(self.cell_idx) > N else self.cell_idx

        for n in idx:
            data = np.memmap(
                self.data_path, dtype='uint8', mode='r', shape=(self.n_genes_original,), offset=n * self.offset
            ).astype(np.float32)
            counts += np.clip(data, 0, 1)
            sums += data

        self.metadata["var"]["mean_expression"] = sums / len(idx)
        self.metadata["var"]["percent_cells"] = 100 * counts / len(idx)

    def _restrict_samples(self, restrictions):

        cond = np.zeros(len(self.metadata["obs"]["SubID"]), dtype=np.uint8)
        cond[self.cell_idx] = 1

        # for AMP-PD, should already be eliminated in the train-test splits
        idx = np.where(np.array(self.metadata["obs"]["SubID"]) == "PM-MS_55245")[0]
        cond[idx] = 0
        print(f"Removing SubID PM-MS_55245, number of samples removed {len(idx)}")

        if restrictions is not None:
            for k, v in restrictions.items():
                if isinstance(v, list):
                    cond *= np.sum(np.stack([np.array(self.metadata["obs"][k]) == v1 for v1 in v]), axis=0).astype(np.uint8)
                else:
                    cond *= np.array(self.metadata["obs"][k]) == v

        self.cell_idx = np.where(cond)[0]
        self.n_samples = len(self.cell_idx)

        for k in self.metadata["obs"].keys():
            self.metadata["obs"][k] = np.array(self.metadata["obs"][k])[self.cell_idx]

        print(f"Restricting samples; number of samples: {self.n_samples}")
        #print(f"Subclasses: {np.unique(self.metadata['obs']['subclass'])}")
        print(f"Number of subjects: {len(np.unique(self.metadata['obs']['SubID']))}")


    def _get_gene_index(self):

        cond = self.metadata["var"]['percent_cells'] >= 0.0

        if self.remove_sex_chrom:
            cond *= self.metadata["var"]['gene_chrom'] != "X"
            cond *= self.metadata["var"]['gene_chrom'] != "Y"
            self.metadata["var"]['percent_cells'][~cond] = 0.0

        if self.protein_coding_only:
            cond *= self.metadata["var"]['protein_coding']
            self.metadata["var"]['percent_cells'][~cond] = 0.0

        if self.top_k_genes is not None:
            th = np.sort(self.metadata["var"]['percent_cells'])[-self.top_k_genes]
            cond *= self.metadata["var"]['percent_cells'] >= th
            print(f"Top {self.top_k_genes} genes selected; threshold = {th:1.4f}")

        self.gene_idx = np.where(cond)[0]

        self.n_genes = len(self.gene_idx)
        self.gene_names = self.metadata["var"]["gene_name"][self.gene_idx]  # needed for pathway networks
        print(f"Sub-sampling genes. Number of genes is now {self.n_genes}")

    def _get_batch_prop_vals(self):

        if self.batch_properties is None:
            return None

        n_batch_properties = len(self.batch_properties.keys())

        self.batch_labels = np.zeros((self.n_samples, n_batch_properties), dtype=np.int64)
        self.batch_mask = np.ones((self.n_samples, n_batch_properties), dtype=np.int64)

        for n0 in range(self.n_samples):
            for n1, (k, prop) in enumerate(self.batch_properties.items()):
                cell_val = self.metadata["obs"][k][n0]
                idx = np.where(cell_val == np.array(prop["values"]))[0]
                # cell property values of -1 will imply N/A, and will be masked out
                if len(idx) == 0:
                    self.batch_labels[n0, n1] = -100
                    self.batch_mask[n0, n1] = 0
                else:
                    self.batch_labels[n0, n1] = idx[0]

    def _get_cell_prop_vals(self):
        """Extract the cell property value for ach entry in the batch"""
        if self.n_cell_properties == 0:
            return None

        self.labels = np.zeros((self.n_samples, self.n_cell_properties), dtype=np.float32)
        self.mask = np.ones((self.n_samples, self.n_cell_properties), dtype=np.float32)
        self.cell_freq = np.ones((self.n_samples,), dtype=np.float32)
        self.subjects = []

        for n0 in range(self.n_samples):

            self.subjects.append(self.metadata["obs"]["SubID"][n0])

            for n1, (k, cell_prop) in enumerate(self.cell_properties.items()):
                cell_val = self.metadata["obs"][k][n0]
                if not cell_prop["discrete"]:
                    # continuous value
                    if cell_val > self.max_cell_prop_val or cell_val < -self.max_cell_prop_val or np.isnan(cell_val):
                        self.labels[n0, n1] = -100
                        self.mask[n0, n1] = 0.0
                    elif "path_braak_lb" in k or "CERAD" in k or "BRAAK_AD" in k:
                        self.labels[n0, n1] = cell_val
                    else:
                        self.labels[n0, n1] = (cell_val - cell_prop["mean"]) / cell_prop["std"]

                else:
                    # discrete value
                    idx = np.where(cell_val == np.array(cell_prop["values"]))[0]
                    # cell property values of -1 will imply N/A, and will be masked out
                    if len(idx) == 0:
                        self.labels[n0, n1] = -100
                        self.mask[n0, n1] = 0.0
                    else:
                        self.labels[n0, n1] = idx[0]

        self.unique_subjects = np.unique(self.subjects)
        self.subjects = np.array(self.subjects)

        print("Finished creating labels")

    def _get_cell_prop_vals_batch(self, batch_idx: List[int]):

        if self.batch_properties is not None:
            return (
                self.labels[batch_idx],
                self.mask[batch_idx],
                self.batch_labels[batch_idx],
                self.batch_mask[batch_idx],
                self.subjects[batch_idx],
            )
        else:
            return self.labels[batch_idx], self.mask[batch_idx], None, None, self.subjects[batch_idx]


    def _get_gene_vals_batch(self, batch_idx: List[int]):

        gene_vals = np.zeros((len(batch_idx), self.n_genes), dtype=np.float32)
        for n, i in enumerate(batch_idx):
            j = self.cell_idx[i]
            gene_vals[n, :] = np.memmap(
                self.data_path, dtype='uint8', mode='r', shape=(self.n_genes_original,), offset=j * self.offset
            )[self.gene_idx].astype(np.float32)
        return gene_vals

    def _prepare_data(self, batch_idx):

        # get input and target data, returned as numpy arrays
        gene_vals = self._get_gene_vals_batch(batch_idx)
        cell_prop_vals, cell_mask, batch_labels, batch_mask, subject = self._get_cell_prop_vals_batch(batch_idx)

        return gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask, subject


    def _get_balanced_batch(self, old_batch_idx):

        batch_idx = []
        group_idx = []
        for k, v in self.group_idx.items():
            replace = False if len(v) >= self.cell_per_group else True
            idx = np.random.choice(v, size=self.cell_per_group, replace=replace)
            batch_idx += idx.tolist()
            group_idx += [k] * self.cell_per_group

        # assert len(batch_idx) == self.cell_per_group * len(self.group_idx), "Batch size not right"

        return batch_idx, group_idx

    def __getitem__(self, batch_idx: Union[int, List[int]]):

        if isinstance(batch_idx, int):
            batch_idx = [batch_idx]

        if self.training and self.group_balancing is not None:
            batch_idx, group_idx = self._get_balanced_batch(batch_idx)
        else:
            group_idx = None

        gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask, subject = self._prepare_data(batch_idx)

        cell_idx = self.cell_idx[batch_idx]


        return (gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask, cell_idx, group_idx, subject)


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
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        remove_sex_chrom: bool = False,
        protein_coding_only: bool = False,
        top_k_genes: Optional[int] = None,
        cell_restrictions: Optional[Dict[str, Any]] = None,
        group_balancing: Literal[None, "bcd", "bc"] = "bd",
        mixup: bool = False,
        cutmix: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.remove_sex_chrom = remove_sex_chrom
        self.protein_coding_only = protein_coding_only
        self.top_k_genes = top_k_genes
        self.cell_restrictions = cell_restrictions
        self.group_balancing = group_balancing
        self.mixup = mixup
        self.cutmix = cutmix
        self._get_cell_prop_info()
        self._get_batch_prop_info()


    def _get_batch_prop_info(self):

        metadata = pickle.load(open(self.metadata_path, "rb"))

        if self.batch_properties is None:
            pass
        else:
            # assuming all batch keys are discrete
            for k in self.batch_properties.keys():
                cell_vals = metadata["obs"][k]

                if self.batch_properties[k]["values"] is None:
                    unique_list = np.unique(cell_vals)
                    # remove nans, negative values, or anything else suspicious
                    idx = [
                        n for n, u in enumerate(unique_list) if (
                                isinstance(u, str) or (u >= 0 and u < 999)
                        )
                    ]
                    print("BATCH VALS",k, unique_list[idx])
                    self.batch_properties[k]["values"] = unique_list[idx]


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

                cell_vals = np.array(metadata["obs"][k])

                if "freq" in self.cell_properties[k] or "mean" in self.cell_properties[k]:
                    continue
                if not cell_prop["discrete"]:

                    idx = [n for n, cv in enumerate(cell_vals) if cv >= -99 and cv < max_cell_prop_val]
                    self.cell_properties[k]["mean"] = np.mean(cell_vals[idx])
                    self.cell_properties[k]["std"] = np.std(cell_vals[idx])
                    print(f"Property: {k}, mean: {self.cell_properties[k]['mean']}, std: {self.cell_properties[k]['std']}")

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
                    print(f"Property: {k}, values: {self.cell_properties[k]['values']}, freq: {self.cell_properties[k]["freq"]}")


                elif cell_prop["discrete"] and cell_prop["values"] is not None:

                    unique_list, counts = np.unique(cell_vals, return_counts=True)
                    idx = [n for n, u in enumerate(unique_list) if u in cell_prop["values"]]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])
                    print(f"Property: {k}, values: {self.cell_properties[k]['values']}, freq: {self.cell_properties[k]["freq"]}")

        else:
            self.cell_prop_dist = None


    def setup(self, stage):

        self.train_dataset = SingleCellDataset(
            self.data_path,
            self.metadata_path,
            self.train_idx,
            cell_properties=self.cell_properties,
            batch_properties=self.batch_properties,
            batch_size=self.batch_size,
            remove_sex_chrom=self.remove_sex_chrom,
            protein_coding_only=self.protein_coding_only,
            top_k_genes=self.top_k_genes,
            cell_restrictions=self.cell_restrictions,
            group_balancing=self.group_balancing,
            training=True,
        )
        self.val_dataset = SingleCellDataset(
            self.data_path,
            self.metadata_path,
            self.test_idx,
            cell_properties=self.cell_properties,
            batch_properties=self.batch_properties,
            batch_size=self.batch_size,
            remove_sex_chrom=self.remove_sex_chrom,
            protein_coding_only=self.protein_coding_only,
            top_k_genes=self.top_k_genes,
            cell_restrictions=self.cell_restrictions,
            gene_idx=self.train_dataset.gene_idx,
            group_balancing=None,
            training=False,
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
            SequentialSampler(self.val_dataset),
            #RandomSampler(self.val_dataset),
            batch_size=self.val_dataset.batch_size,
            drop_last=False,
        )
        dl = DataLoader(
            self.val_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=self.num_workers,
        )
        return dl
