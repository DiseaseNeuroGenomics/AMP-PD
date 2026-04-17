from typing import Any, Dict, List, Optional

import os, shutil
import copy
import math
import pickle
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from torch.distributions import NegativeBinomial, Bernoulli, Normal, kl_divergence
from torchmetrics import MetricCollection, ExplainedVariance, MeanSquaredError
from torchmetrics.classification import Accuracy, BinaryAccuracy
import torch.nn.functional as F
from torch.distributions.normal import Normal
from losses import ZINB

from transformers import get_cosine_schedule_with_warmup


class BaseTask(pl.LightningModule):

    def __init__(
            self,
            network,
            task_cfg,
            **kwargs
    ):

        # Initialize superclass
        super().__init__()
        self.network = network
        self.task_cfg = task_cfg
        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None
        self.source_code_copied = False

    def _copy_source_code(self):

        try:
            target_dir = f"{self.trainer.log_dir}/code"
            os.mkdir(target_dir)
            base_dir = os.getcwd()
            print(f"Base dir: {base_dir}")
            src_files = [
                f"{base_dir}/config.py",
                f"{base_dir}/config_tf.py",
                f"{base_dir}/datasets.py",
                f"{base_dir}/networks.py",
                f"{base_dir}/modules.py",
                f"{base_dir}/tasks.py",
                f"{base_dir}/train.py",
            ]
            for src in src_files:
                shutil.copyfile(src, f"{target_dir}/{os.path.basename(src)}")
        except:
            return

    def training_step(self, batch, batch_idx):

        return None

    def validation_step(self, batch, batch_idx):

        return None

    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        fn = f"{self.trainer.log_dir}/test_results.pkl"
        pickle.dump(self.results, open(fn, "wb"))
        self.results["epoch"] = self.current_epoch + 1

    def configure_optimizers(self):

        return torch.optim.AdamW(
            self.network.parameters(),
            lr=self.task_cfg["learning_rate"],
            weight_decay=self.task_cfg["weight_decay"],
            betas=self.task_cfg["ADAM_betas"],
            eps=self.task_cfg["ADAM_eps"],
        )

    def optimizer_step(
            self,
            current_epoch,
            batch_nb,
            optimizer,
            closure,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
            min_lr=1e-8,
    ):

        # warm up lr
        if self.trainer.global_step < self.task_cfg["warmup_steps"]:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.task_cfg["warmup_steps"]))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.task_cfg["learning_rate"]


        elif self.trainer.global_step > self.task_cfg["warmup_steps"]:
            lr_scale = self.task_cfg["decay"] ** (self.trainer.global_step - self.task_cfg["warmup_steps"])
            lr_scale = max(min_lr, lr_scale)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.task_cfg["learning_rate"]

        # update params
        optimizer.step(closure=closure)


class RecursiveInference(BaseTask):
    def __init__(
            self,
            network,
            task_cfg,
            **kwargs
    ):

        # Initialize superclass
        super().__init__(network, task_cfg)

        for n, p in self.network.named_parameters():
            if not "feature" in n:
                p.requires_grad_(False)

        # self._load_gene_perturb_info()
        self._create_results_dict()
        self.source_code_copied = False
        self.iterations_per_gene = 50

    def _load_gene_perturb_info(self):
        self.gene_perturb = self.task_cfg["gene_perturb"]
        self.gene_perturb_idx = np.where(np.array(self.gene_names) == self.gene_perturb)[0][0]

    def _create_results_dict(self):

        self.n_genes = len(self.gene_names)
        n_cells = len(self.task_cfg["cell_idx"])
        self.perturb_gene_count = -1
        self.validation_count = -1
        self.perturb_gene_list = self.task_cfg["perturb_genes"]
        self.actual_exp = []
        self.pred_exp = []

        self.results = {
            "gene_names": self.gene_names,
            # "gene_perturb": self.gene_perturb,
            # "gene_perturb_idx": self.gene_perturb_idx,
            "actual_exp": [],
            "pred_exp": {k: [] for k in range(50)},
            "perturb_gene": [],
            "counts": [],
        }

    def _subsample_batch(self, batch):

        gene_ids, gene_target_ids, gene_vals, gene_targets, depths, _, _, cell_idx = batch

        # search for out perturb in the input
        gene_ids_with_exp = gene_ids * (
                gene_vals >= math.log(1 + self.task_cfg["min_count_gene_expression"])
        ).to(torch.int64)

        perturb_idx = torch.where(gene_ids_with_exp == self.gene_perturb_idx)
        if len(perturb_idx[0]) < 2:
            # too few samples, don't both
            return None, None

        batch = (
            gene_ids[perturb_idx[0]],
            gene_target_ids[perturb_idx[0]],
            gene_vals[perturb_idx[0]],
            gene_targets[perturb_idx[0]],
            # key_padding_mask[perturb_idx[0]],
            depths[perturb_idx[0]] if depths is not None else None,
            [cell_idx[i] for i in perturb_idx[0]],
        )

        perturb_idx = torch.where(batch[0] == self.gene_perturb_idx)
        return batch, perturb_idx

    def _get_possible_gene_index(self, batch):

        gene_ids = batch[0].detach().cpu().numpy()
        gene_target_ids = batch[1].detach().cpu().numpy()
        idx = list(gene_ids.flatten()) + list(gene_target_ids.flatten())
        return np.unique(idx)

    def on_validation_epoch_start(self):

        # reset after validation check
        if self.validation_count == 1:
            self.validation_count = -1
            self.perturb_gene_count = -1
            self.actual_exp = []
            for k in range(50):
                self.pred_exp[k] = []

    def validation_stepX(self, batch, batch_idx):

        n_recursive_steps = 1
        n_genes = len(self.gene_names)
        self.validation_count += 1

        if self.validation_count == 0:
            self.all_possible_gene_idx = self._get_possible_gene_index(batch)
            self.gene_position = {}
            for i, n in enumerate(self.all_possible_gene_idx):
                self.gene_position[n] = i

        if self.validation_count % self.iterations_per_gene == 0:
            self.perturb_gene_count += 1
            self.perturb_gene = self.perturb_gene_list[self.perturb_gene_count]
            self.gene_perturb_idx = np.where(np.array(self.gene_names) == self.perturb_gene)[0][0]
            self.possible_gene_idx = list(set(self.all_possible_gene_idx) - set([self.gene_perturb_idx]))
            self.actual_exp = []
            self.pred_exp = {k: [] for k in range(n_recursive_steps)}

        batch, perturb_idx = self._subsample_batch(batch)

        k = 0

        if batch is not None:  # happens where there are too few samples containing the gene to perturb

            gene_ids, old_gene_target_ids, gene_vals, gene_targets, depths, cell_idx = batch
            batch_size, n_input_genes = gene_ids.shape

            for i in range(batch_size):
                vals = np.zeros(n_genes, dtype=np.float32)
                gene_idx = old_gene_target_ids[i, :].detach().to(torch.int64).cpu().numpy()
                vals[gene_idx] = gene_targets[i, :].detach().to(torch.float32).cpu().numpy()
                self.actual_exp.append(vals)

            gene_target_ids = torch.tile(torch.arange(self.n_genes)[None, :], (batch_size, 1)).to(
                old_gene_target_ids.device)

            gene_pred, _, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals, None, depths,
            )

            if self.task_cfg["perturb_knockdown"]:
                perturb_gene_original = gene_vals[perturb_idx]
                perturb_gene_clamp_val = 0.0
            else:
                perturb_gene_clamp_val = gene_vals[perturb_idx] - 1

            gene_vals_perturb = gene_vals
            gene_vals_perturb[perturb_idx] = copy.deepcopy(perturb_gene_clamp_val)

            gene_pred_perturb, _, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals_perturb, None, depths,
            )

            delta = gene_pred_perturb[:, :, 0] - gene_pred[:, :, 0]
            #print("A", torch.abs(delta).mean())
            new_gene_vals = gene_targets[:, :] + delta[:, self.all_possible_gene_idx]



            ###################

            #gene_pred = copy.deepcopy(gene_pred_perturb)
            gene_vals = copy.deepcopy(new_gene_vals)
            gene_vals_perturb = copy.deepcopy(new_gene_vals)
            gene_ids = gene_target_ids[:, self.all_possible_gene_idx]
            perturb_idx = torch.where(gene_ids == self.gene_perturb_idx)

            gene_vals_perturb[perturb_idx] = copy.deepcopy(perturb_gene_clamp_val)

            gene_pred, _, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals, None, depths,
            )

            gene_pred_perturb, _, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals_perturb, None, depths,
            )

            delta = gene_pred_perturb[:, :, 0] - gene_pred[:, :, 0]
            #print("B", torch.abs(delta).mean())
            new_gene_vals = gene_targets[:, :] + delta[:, self.all_possible_gene_idx]

            ####

            for i in range(batch_size):
                vals = np.zeros(n_genes, dtype=np.float32)
                vals[self.all_possible_gene_idx] = new_gene_vals[i, :].detach().to(torch.float32).cpu().numpy()
                self.pred_exp[k].append(vals)

        if self.validation_count % self.iterations_per_gene == self.iterations_per_gene - 1:

            # save previous data
            if len(self.actual_exp) > 0:
                self.actual_exp = np.stack(self.actual_exp)

                self.results["actual_exp"].append(np.mean(self.actual_exp, axis=0))
                for k in range(n_recursive_steps):
                    self.pred_exp[k] = np.stack(self.pred_exp[k])
                    self.results["pred_exp"][k].append(np.mean(self.pred_exp[k], axis=0))
                self.results["perturb_gene"].append(self.perturb_gene)
                self.results["counts"].append(self.pred_exp[k].shape[0])

    def validation_step(self, batch, batch_idx):

        n_recursive_steps = 10
        n_genes = len(self.gene_names)
        self.validation_count += 1

        if self.validation_count == 0:
            self.all_possible_gene_idx = self._get_possible_gene_index(batch)
            self.gene_position = {}
            for i, n in enumerate(self.all_possible_gene_idx):
                self.gene_position[n] = i

        if self.validation_count % self.iterations_per_gene == 0:
            self.perturb_gene_count += 1
            self.perturb_gene = self.perturb_gene_list[self.perturb_gene_count]
            self.gene_perturb_idx = np.where(np.array(self.gene_names) == self.perturb_gene)[0][0]
            self.possible_gene_idx = list(set(self.all_possible_gene_idx) - set([self.gene_perturb_idx]))
            self.actual_exp = []
            self.pred_exp = {k: [] for k in range(n_recursive_steps)}


        batch, perturb_idx = self._subsample_batch(batch)



        if batch is not None:  # happens where there are too few samples containing the gene to perturb

            gene_ids, old_gene_target_ids, gene_vals, gene_targets, depths, cell_idx = batch
            batch_size, n_input_genes = gene_ids.shape


            gene_target_ids = torch.tile(torch.arange(self.n_genes)[None, :], (batch_size, 1)).to(
                old_gene_target_ids.device)

            gene_pred, _, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals, None, depths,
            )

            #print("AAAA", gene_pred.shape, gene_vals.shape, gene_pred.mean(), gene_vals.mean())

            prev_gene_pred = copy.deepcopy(gene_pred[..., 0])


            if self.task_cfg["perturb_knockdown"]:
                perturb_gene_original = gene_vals[perturb_idx]
                perturb_gene_clamp_val = 0.0
            else:
                gene_val_counts = torch.exp(gene_vals[perturb_idx]) - 1

                perturb_gene_clamp_val = torch.log(1 + gene_val_counts + self.task_cfg["perturb_val"])

            # experimental, remove 1 count from the depths
            depths = torch.log(torch.exp(depths) - 1)

            prev_gene_vals = torch.zeros((batch_size, self.n_genes), dtype=torch.float32).to(gene_vals.device)
            for i in range(batch_size):
                vals = np.zeros(n_genes, dtype=np.float32)
                gene_idx = old_gene_target_ids[i, :].detach().to(torch.int64).cpu().numpy()
                vals[gene_idx] = gene_targets[i, :].detach().to(torch.float32).cpu().numpy()
                prev_gene_vals[i, old_gene_target_ids[i, :]] = gene_targets[i, :]
                self.actual_exp.append(vals)


            gene_vals_perturb = gene_vals
            gene_vals_perturb[perturb_idx] = copy.deepcopy(perturb_gene_clamp_val)

            pi_pred = gene_pred[:, :, 1]
            #print("INPUT", gene_targets.mean())

            for k in range(n_recursive_steps):

                if self.task_cfg["loss"] == "ZINB" or self.task_cfg["loss"] == "MSE":


                    samples, prev_gene_pred = self.network.generate_samples(
                        gene_ids,
                        gene_target_ids,
                        gene_vals_perturb,
                        depths,
                        pi_pred=None,
                        loss="ZINB",
                        prev_gene_vals=prev_gene_vals,
                        prev_gene_pred=prev_gene_pred,
                    )

                    samples = torch.clamp(samples, min=0, max=254)
                    prev_gene_vals = copy.deepcopy(samples)

                    for i in range(batch_size):
                        samples[i, gene_ids[i, :]] = torch.exp(copy.deepcopy(gene_vals_perturb[i, :])) - 1

                    gene_pred_perturb = torch.log(1 + samples).to(torch.float32)
                    #gene_pred_perturb = copy.deepcopy(samples)
                    #pi_pred = copy.deepcopy(pi)
                    #print("mean samples", gene_pred_perturb.mean())
                    depths = None
                    if depths is not None:
                        depth = samples[:, self.all_possible_gene_idx].sum(dim=1)
                        # depth = (torch.exp(samples[:, self.all_possible_gene_idx]) - 1).sum(dim=1)
                        # depths[:, 0] = torch.log(1 + depth)
                        # depths[:, 1] = torch.log(1 + depth)

                else:
                    gene_pred_perturb = torch.clamp(gene_pred_perturb[..., 0].to(torch.float32), min=0.0)



                # print("XX", gene_vals_perturb.shape, perturb_gene_clamp_val.shape)
                gene_ids[:, 0] = self.gene_perturb_idx
                gene_vals_perturb[:, 0] = copy.deepcopy(perturb_gene_clamp_val)


                for i in range(batch_size):
                    idx_input = np.random.choice(self.possible_gene_idx, n_input_genes - 1, replace=False)
                    gene_ids[i, 1:] = torch.from_numpy(idx_input).to(gene_ids.device)
                    # idx_input_vals = [self.gene_position[j] for j in idx_input]
                    # gene_vals_perturb[i, 1:] = copy.deepcopy(gene_pred_perturb[i, idx_input_vals])
                    gene_vals_perturb[i, 1:] = copy.deepcopy(gene_pred_perturb[i, gene_ids[i, 1:]])

                    vals = np.zeros(n_genes, dtype=np.float32)
                    vals[self.all_possible_gene_idx] = gene_pred_perturb[i, self.all_possible_gene_idx].detach().to(
                        torch.float32).cpu().numpy()
                    self.pred_exp[k].append(vals)

        if self.validation_count % self.iterations_per_gene == self.iterations_per_gene - 1:

            # save previous data
            if len(self.actual_exp) > 0:
                self.actual_exp = np.stack(self.actual_exp)

                self.results["actual_exp"].append(np.mean(self.actual_exp, axis=0))
                for k in range(n_recursive_steps):
                    self.pred_exp[k] = np.stack(self.pred_exp[k])
                    self.results["pred_exp"][k].append(np.mean(self.pred_exp[k], axis=0))
                self.results["perturb_gene"].append(self.perturb_gene)
                self.results["counts"].append(self.pred_exp[k].shape[0])


class TFInference(BaseTask):
    def __init__(
            self,
            network,
            task_cfg,
            **kwargs
    ):

        # Initialize superclass
        super().__init__(network, task_cfg)

        for n, p in self.network.named_parameters():
            if not "feature" in n:
                p.requires_grad_(False)

        self._load_tf_info()
        self._create_results_dict()
        self.source_code_copied = False

    def _load_tf_info(self):
        self.tf_list = pd.read_csv(self.task_cfg["tf_list_fn"], header=None)[0].values
        self.tf_list_idx = []
        for tf in self.tf_list:
            if tf in self.gene_names:
                self.tf_list_idx.append(np.where(np.array(self.gene_names) == tf)[0][0])

    def _create_results_dict(self):

        n_tfs = len(self.tf_list)
        n_cells = len(self.task_cfg["cell_idx"])
        self.results = {
            "gene_names": self.gene_names,
            "tf_list": self.tf_list,
            "tf_list_idx": self.tf_list_idx,
            "cell_idx": self.task_cfg["cell_idx"],
            "counts": np.zeros((n_cells, n_tfs), dtype=np.int16),
            "pred_tf_exp": np.zeros((n_cells, n_tfs), dtype=np.float32),
            "actual_tf_exp": np.zeros((n_cells, n_tfs), dtype=np.float32),
        }

    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask, depths, _, _, cell_idx = batch
        batch_size, n_target_genes = gene_target_ids.shape

        gene_pred, _, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask, depths,
        )

        gene_target_ids = gene_target_ids.detach().cpu().numpy()
        gene_target_ids = np.reshape(gene_target_ids, (-1))
        gene_pred = gene_pred.to(torch.float32).detach().cpu().numpy()
        gene_pred = np.reshape(gene_pred, (-1))

        gene_targets = gene_targets.to(torch.float32).detach().cpu().numpy()
        gene_targets = np.reshape(gene_targets, (-1))

        cell_idx = np.array(cell_idx)
        cell_idx = np.tile(cell_idx[:, None], (1, n_target_genes))
        cell_idx = np.reshape(cell_idx, (-1))

        for j, i in enumerate(self.tf_list_idx):
            if i in gene_target_ids:
                idx = np.where(gene_target_ids == i)[0]
                rows = cell_idx[idx]
                # cols = gene_target_ids[idx]
                self.results["counts"][rows, j] += 1
                self.results["pred_tf_exp"][rows, j] += gene_pred[idx]
                self.results["actual_tf_exp"][rows, j] = gene_targets[idx]


class AllToAllPerutbation(BaseTask):

    def __init__(
            self,
            network,
            task_cfg,
            **kwargs
    ):
        # Initialize superclass
        super().__init__(network, task_cfg)

        for n, p in self.network.named_parameters():
            p.requires_grad_(False)

        self._load_tf_info() # not strictly needed
        self._create_results_dict()


    def _load_tf_info(self):
        self.tf_list = pd.read_csv(self.task_cfg["tf_list_fn"], header=None)[0].values
        self.tf_list_idx = []
        for tf in self.tf_list:
            if tf in self.gene_names:
                self.tf_list_idx.append(np.where(np.array(self.gene_names) == tf)[0][0])

    def _create_results_dict(self):

        self.results = {"gene_names": self.gene_names, "tf_list": self.tf_list}
        N = len(self.gene_names)

        self.results["pos_delta"] = np.zeros((N, N), dtype=np.float32)
        self.results["pos_var"] = np.zeros((N, N), dtype=np.float32)
        self.results["pos_count"] = np.zeros((N, N), dtype=np.uint16)

        # self.results["neg_delta"] = np.zeros((N, N), dtype=np.float32)
        # self.results["neg_var"] = np.zeros((N, N), dtype=np.float32)
        # self.results["neg_count"] = np.zeros((N, N), dtype=np.uint16)

    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        fn = f"{self.trainer.log_dir}/test_results.pkl"
        if (self.current_epoch + 1) % 25 == 0:
            print("Saving data")
            pickle.dump(self.results, open(fn, "wb"))
        self.results["epoch"] = self.current_epoch + 1

    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, padding_mask, target_mask, depths, _, _, _ = batch

        batch_size = gene_ids.shape[0]
        n_iters = 10

        n_input_genes = gene_vals.shape[1]

        gene_pred, _, _, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, None, depths,
        )

        target_idx = gene_target_ids.to(torch.int16).detach().cpu().numpy()
        rnd_idx = np.random.choice(n_input_genes, size=(10,), replace=False)

        for n in range(n_iters):

            # rnd_inc = torch.from_numpy(np.random.choice([1, 2, 3], size=(batch_size,), replace=True)).to(gene_vals.device)

            gene_vals_perturb = copy.deepcopy(gene_vals.clone().detach())
            # gene_vals_perturb[:, rnd_idx[n]] = torch.log(1 + torch.exp(gene_vals_perturb[:, rnd_idx[n]]))
            gene_vals_perturb[:, rnd_idx[n]] = gene_vals_perturb[:, rnd_idx[n]] + 1
            # TODO: add fix when binning inputs
            # gene_vals_perturb[:, rnd_idx[n]] = torch.clip(
            #    gene_vals_perturb[:, rnd_idx[n]] + rnd_inc, 0, self.n_bins-1
            # )

            gene_pred_perturb, _, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals_perturb, None, depths,
            )

            source_idx = gene_ids[:, rnd_idx[n]].to(torch.int16).detach().cpu().numpy()
            delta = gene_pred_perturb.to(torch.float32).detach().cpu().numpy() - gene_pred.to(
                torch.float32).detach().cpu().numpy()

            delta = delta[:, :, 2]

            for i in range(batch_size):
                count = self.results["pos_count"][source_idx[i], target_idx[i]]
                prev_mean = self.results["pos_delta"][source_idx[i], target_idx[i]] / (1e-6 + count)
                prev_var = self.results["pos_var"][source_idx[i], target_idx[i]]

                self.results["pos_count"][source_idx[i], target_idx[i]] += 1
                self.results["pos_delta"][source_idx[i], target_idx[i]] += delta[i, :]

                count = self.results["pos_count"][source_idx[i], target_idx[i]]
                current_mean = self.results["pos_delta"][source_idx[i], target_idx[i]] / (1e-6 + count)

                current_var = prev_var + (delta[i, :] - prev_mean) * (delta[i, :] - current_mean)
                self.results["pos_var"][source_idx[i], target_idx[i]] = current_var


class TFoAllPerutbation(BaseTask):

    def __init__(
            self,
            network,
            task_cfg,
            **kwargs
    ):
        # Initialize superclass
        super().__init__(network, task_cfg)

        for n, p in self.network.named_parameters():
            p.requires_grad_(False)

        self._load_tf_info()
        self._create_results_dict()


    def _load_tf_info(self):
        self.tf_list = pd.read_csv(self.task_cfg["tf_list_fn"], header=None)[0].values
        self.tf_list_idx = []
        for tf in self.tf_list:
            if tf in self.gene_names:
                self.tf_list_idx.append(np.where(np.array(self.gene_names) == tf)[0][0])

        print(f"Length of tf list: {len(self.tf_list_idx)}")

    def _create_results_dict(self):

        self.results = {"gene_names": self.gene_names, "tf_list": self.tf_list}
        N = len(self.gene_names)
        n_tfs = len(self.tf_list)
        self.results["pos_delta"] = np.zeros((n_tfs, N), dtype=np.float32)
        self.results["pos_var"] = np.zeros((n_tfs, N), dtype=np.float32)
        self.results["pos_count"] = np.zeros((n_tfs, N), dtype=np.uint16)

        # self.results["neg_delta"] = np.zeros((N, N), dtype=np.float32)
        # self.results["neg_var"] = np.zeros((N, N), dtype=np.float32)
        # self.results["neg_count"] = np.zeros((N, N), dtype=np.uint16)

    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, depths, _, _, _ = batch
        batch_size = gene_ids.shape[0]
        n_iters = 10

        n_input_genes = gene_vals.shape[1]

        gene_pred, _, _, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, None, depths,
        )

        target_idx = gene_target_ids.to(torch.int16).detach().cpu().numpy()
        rnd_idx = np.random.choice(n_input_genes, size=(10,), replace=False)

        tf_set_idx = set(self.tf_list_idx)

        for n in range(n_iters):

            # rnd_inc = torch.from_numpy(np.random.choice([1, 2, 3], size=(batch_size,), replace=True)).to(gene_vals.device)

            gene_vals_perturb = copy.deepcopy(gene_vals.clone().detach())

            source_idx = np.zeros(batch_size, dtype=np.int64)
            #source_gene_idx = np.zeros(batch_size, dtype=np.int64)

            for i in range(batch_size):
                gene_ids_numpy = gene_ids[i, :].cpu().numpy()
                idx_intersection = list(tf_set_idx.intersection(set(gene_ids_numpy)))
                j = np.random.choice(idx_intersection)
                k = np.where(gene_ids_numpy == j)[0]
                source_idx[i] = np.where(np.array(self.tf_list_idx) == j)[0][0]
                #gene_vals_perturb[i, k] = torch.log(1 + torch.exp(gene_vals_perturb[i, k]))
                gene_vals_perturb[i, k] += 1.0


            # gene_vals_perturb[:, rnd_idx[n]] = torch.log(1 + torch.exp(gene_vals_perturb[:, rnd_idx[n]]))
            # TODO: add fix when binning inputs
            # gene_vals_perturb[:, rnd_idx[n]] = torch.clip(
            #    gene_vals_perturb[:, rnd_idx[n]] + rnd_inc, 0, self.n_bins-1
            # )

            gene_pred_perturb, _, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals_perturb, None, depths,
            )

            # source_idx = gene_ids[:, rnd_idx[n]].to(torch.int16).detach().cpu().numpy()
            # delta = gene_pred_perturb.to(torch.float32).detach().cpu().numpy() - gene_pred.to(
            #    torch.float32).detach().cpu().numpy()

            #delta0 = (
            #    torch.exp(gene_pred_perturb[..., 0]) -  torch.exp(gene_pred[..., 0])
            #).to(torch.float32).detach().cpu().numpy()

            delta = (gene_pred_perturb[..., 0] - gene_pred[..., 0]).to(torch.float32).detach().cpu().numpy()


            for i in range(batch_size):
                count = self.results["pos_count"][source_idx[i], target_idx[i]]
                prev_mean = self.results["pos_delta"][source_idx[i], target_idx[i]] / (1e-6 + count)
                prev_var = self.results["pos_var"][source_idx[i], target_idx[i]]

                self.results["pos_count"][source_idx[i], target_idx[i]] += 1
                self.results["pos_delta"][source_idx[i], target_idx[i]] += delta[i, :]
                #self.results["pos_delta"][source_idx[i], target_idx[i], 1] += delta1[i, :]

                count = self.results["pos_count"][source_idx[i], target_idx[i]]
                current_mean = self.results["pos_delta"][source_idx[i], target_idx[i]] / (1e-6 + count)

                current_var = prev_var + (delta[i, :] - prev_mean) * (delta[i, :] - current_mean)
                self.results["pos_var"][source_idx[i], target_idx[i]] = current_var

class VAELoss(BaseTask):
    def __init__(
            self,
            network,
            task_cfg,
            **kwargs
    ):
        # Initialize superclass
        super().__init__(network, task_cfg)

        self.cell_properties = task_cfg["cell_properties"]
        self.zinb = ZINB(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.val_mse = nn.MSELoss(reduction="none")
        self.explained_var = ExplainedVariance()
        self.explained_var_mse = ExplainedVariance()

        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None
        self._create_results_dict()
        self._cell_properties_metrics()
        self.source_code_copied = False


        for k, v in self.task_cfg.items():
            print(k, v)

    def _cell_properties_metrics(self):

        """Used to calculate validation metrics, and to save model predictions"""
        self.cell_cross_ent = nn.ModuleDict()
        self.cell_mse = nn.ModuleDict()
        self.cell_accuracy = nn.ModuleDict()
        self.cell_explained_var = nn.ModuleDict()

        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                # discrete variable, set up cross entropy module

                if self.task_cfg["focal_loss"]:
                    pass
                    """
                    self.cell_cross_ent[k] = FocalLoss(
                        len(cell_prop["values"]),
                        gamma=self.focal_loss_gamma,
                        alpha=2.0,
                    )
                    """
                else:
                    self.cell_cross_ent[k] = nn.CrossEntropyLoss(
                        weight=None,
                        reduction="none",
                        ignore_index=-100,
                        # label_smoothing=self.label_smoothing,
                    )

                self.cell_accuracy[k] = Accuracy(
                    task="multiclass", num_classes=cell_prop["n_values"], average="macro",
                )
            else:
                # continuous variable, set up MSE module
                self.cell_mse[k] = nn.MSELoss(reduction="none")
                self.cell_explained_var[k] = ExplainedVariance()

    def _create_results_dict(self):

        self.results = {"epoch": 0, "gene_names": self.gene_names, "cell_properties": {}}
        for k, v in self.cell_properties.items():
            self.results["cell_properties"][k] = v
        self.results_list = ["gene_target_ids", "gene_targets", "gene_pred"]
        for k in self.results_list:
            self.results[k] = []

    def _cell_loss(self, cell_pred: Dict[str, torch.Tensor], targets: torch.Tensor):

        cell_loss = 0.0

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            mask = (targets[:, n] != -100).to(torch.float32)

            if cell_prop["discrete"]:
                loss = self.cell_cross_ent[k](cell_pred[k], targets[:, n].to(torch.int64))
                masked_loss = (loss * mask).mean()
                cell_loss += masked_loss

            else:
                loss = self.cell_mse[k](torch.squeeze(cell_pred[k]), targets[:, n])
                masked_loss = (loss * mask).mean()
                cell_loss += masked_loss

            self.log(f"loss_{k}", masked_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)


        return cell_loss

    def _cell_scores(self, cell_pred: Dict[str, torch.Tensor], targets: torch.Tensor):

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            mask_idx = targets[:, n] != -100

            if cell_prop["discrete"]:
                pred_idx = torch.argmax(cell_pred[k], dim=-1).to(torch.int64)
                self.cell_accuracy[k].update(pred_idx[mask_idx], targets[mask_idx, n].to(torch.int64))

            else:
                pred = cell_pred[k][:, 0]
                #try:  # rare error
                self.cell_explained_var[k].update(pred[mask_idx], targets[mask_idx, n])
                #except:
                #    self.cell_explained_var[k].update(pred, cell_targets[idx, n])

        for k, v in self.cell_accuracy.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.cell_explained_var.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):

        _, _, gene_vals, gene_targets, _, target_mask, depths, cell_prop_vals, batch_prop_vals, _ = batch
        px_mu, px_mu_mse, px_drop, px_theta, qz_m, qz_v, cell_pred = self.network.forward(gene_vals, depths, batch_prop_vals)

        gene_loss = self.zinb(
            torch.exp(gene_targets) - 1,
            torch.exp(px_mu),
            F.softplus(px_theta),
            px_drop,
        )
        gene_loss = (gene_loss * target_mask).sum() / target_mask.sum()

        # KL Divergence
        kl_loss = kl_divergence(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(torch.zeros_like(qz_m),  torch.ones_like(qz_v))
        ).sum(dim=1).mean()

        mse_loss = self.mse(px_mu_mse, gene_targets)
        mse_loss = (mse_loss * target_mask).sum() / target_mask.sum()

        cell_loss = self._cell_loss(cell_pred, cell_prop_vals)

        loss = (
            gene_loss +
            self.task_cfg["kl_loss_coef"] * kl_loss +
            self.task_cfg["mse_loss_coef"] * mse_loss +
            self.task_cfg["cell_loss_coef"] * cell_loss
        )

        #pi_prob = (1 - torch.sigmoid(px_drop))
        #y = torch.exp(px_mu) * pi_prob
        #ex_var_zinb = self.ex_var_zinb(torch.log(1 + y), gene_targets)

        #self.log("ev_zinb", ex_var_zinb, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        _, _, gene_vals, gene_targets, _, target_mask, depths, cell_prop_vals, batch_prop_vals, _ = batch
        px_mu, px_mu_mse, px_drop, px_theta, qz_m, qz_v, cell_pred = self.network.forward(gene_vals, depths, batch_prop_vals)

        #print("gene_vals", gene_vals.mean())
        #print("gene_targets", gene_targets.mean())
        #print("target_mask", target_mask.mean())

        gene_loss = self.zinb(
            torch.exp(gene_targets) - 1,
            torch.exp(px_mu),
            F.softplus(px_theta),
            px_drop,
        )
        gene_loss = (gene_loss * target_mask).sum() / target_mask.sum()

        self._cell_scores(cell_pred, cell_prop_vals)

        pi_prob = (1 - torch.sigmoid(px_drop))
        y = torch.exp(px_mu) * pi_prob

        #print("A", px_mu_mse.shape, target_mask.shape, px_mu_mse.mean(), target_mask.mean())

        y = torch.flatten(y)
        px_mu_mse = torch.flatten(px_mu_mse)
        gene_targets = torch.flatten(gene_targets)
        target_mask = torch.flatten(target_mask)

        #print("B", px_mu_mse.shape, target_mask.shape, px_mu_mse.mean(), target_mask.mean())

        y = y[target_mask.to(torch.bool)]
        px_mu_mse = px_mu_mse[target_mask.to(torch.bool)]
        gene_targets = gene_targets[target_mask.to(torch.bool)]

        #print("C", px_mu_mse.shape, target_mask.shape, px_mu_mse.mean(), target_mask.mean())

        self.explained_var(torch.log(1 + y), gene_targets)
        self.explained_var_mse(px_mu_mse, gene_targets)

        self.log("exp_var", self.explained_var, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("exp_var_mse", self.explained_var_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("gene_loss_val", gene_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)



    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        fn = f"{self.trainer.log_dir}/test_results.pkl"
        pickle.dump(self.results, open(fn, "wb"))
        self.results["epoch"] = self.current_epoch + 1
        for k in self.results_list:
            self.results[k] = []


class MSELoss(BaseTask):
    def __init__(
            self,
            network,
            task_cfg,
            **kwargs
    ):
        # Initialize superclass
        super().__init__(network, task_cfg)

        # Functions and metrics
        if self.task_cfg["loss"] == "MSE":
            self.mse = nn.MSELoss(reduction="none")
        else:
            self.zinb = ZINB()
        self.val_mse_masked = MeanSquaredError()
        self.explained_var_masked = ExplainedVariance()
        self.val_mse_unmasked = MeanSquaredError()
        self.explained_var_unmasked = ExplainedVariance()

        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None
        self._create_results_dict()
        self.source_code_copied = False

    def _create_results_dict(self):

        self.results = {"epoch": 0, "gene_names": self.gene_names}
        self.results_list = ["gene_target_ids", "gene_targets", "gene_pred"]
        for k in self.results_list:
            self.results[k] = []

    def training_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask, target_mask, depths, _, _, _ = batch
        gene_pred, latent, _, _, cell_pred = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask, depths,
        )


        if self.task_cfg["loss"] == "MSE":
            if self.task_cfg["masked_gene_selection"] == "BERT":
                idx_masked = torch.where(target_mask * key_padding_mask)
                idx_unmasked = torch.where(target_mask * (1 - key_padding_mask))
                gene_loss_masked = (self.mse(gene_pred[idx_masked], gene_targets[idx_masked].unsqueeze(-1))).mean()
                gene_loss_unmasked = (
                    self.mse(gene_pred[idx_unmasked], gene_targets[idx_unmasked].unsqueeze(-1))).mean()
                self.log("gene_loss_masked", gene_loss_masked, on_step=True, on_epoch=True, prog_bar=True,
                         sync_dist=True)
                self.log("gene_loss_unmasked", gene_loss_unmasked, on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=True)
                gene_loss = 2 * gene_loss_masked + gene_loss_unmasked
            else:
                gene_loss = (self.mse(gene_pred, gene_targets.unsqueeze(-1))).mean()
                self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

                idx_unmasked = (gene_target_ids.unsqueeze(2) == gene_ids.unsqueeze(1)).any(dim=2)
                idx_masked = (gene_target_ids.unsqueeze(2) != gene_ids.unsqueeze(1)).all(dim=2)
                print("SS", idx_unmasked.sum(), idx_masked.sum())
                idx_m = gene_target_ids[0, idx_masked[0]].cpu().numpy()
                idx_um = gene_target_ids[0, idx_unmasked[0]].cpu().numpy()
                inputs = gene_ids[0, :].cpu().numpy()
                m0 = set(idx_m).intersection(set(inputs))
                m1 = set(idx_um).intersection(set(inputs))
                print("MASK", len(m0), len(inputs), len(idx_m))
                print("UNMASK", len(m1), len(inputs), len(idx_um))


        elif self.task_cfg["loss"] == "ZINB":
            theta = self.network.output_theta(gene_target_ids)
            gene_loss = self.zinb(
                torch.exp(gene_targets) - 1,
                torch.exp(gene_pred[..., 0]),
                # F.softplus(gene_pred[..., 1]),
                # gene_pred[..., 2],
                F.softplus(theta[..., 0]),
                gene_pred[..., 1],
            )

            gene_loss_mse = self.mse(gene_pred[..., 2], gene_targets)
            gene_loss = gene_loss + gene_loss_mse

            pi_prob = (1 - torch.sigmoid(gene_pred[..., 1]))
            y = torch.exp(gene_pred[..., 0]) * pi_prob
            ex_var_zinb = self.ex_var_zinb(torch.log(1 + y), gene_targets)
            ex_var_mse = self.ex_var_mse(gene_pred[..., 2], gene_targets)

            self.log("ev_zinb", ex_var_zinb, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("ev_mse", ex_var_mse, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            print("XXXXX")

        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return gene_loss

    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask, target_mask, depths, _, _, _ = batch

        gene_pred, _, _, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask, depths,
        )

        if self.task_cfg["save_predictions"]:
            self.results["gene_target_ids"].append(gene_target_ids.to(torch.int16).detach().cpu().numpy())
            self.results["gene_targets"].append(gene_targets.detach().cpu().numpy())
            self.results["gene_pred"].append(gene_pred.to(torch.float32).detach().cpu().numpy())

        if self.task_cfg["loss"] == "MSE":
            if self.task_cfg["masked_gene_selection"] == "BERT":
                idx_masked = torch.where(target_mask * key_padding_mask)
                idx_unmasked = torch.where(target_mask * (1 - key_padding_mask))
            else:
                # idx_unmasked =  (gene_ids.unsqueeze(2) == gene_target_ids.unsqueeze(1)).any(dim=2)
                # idx_masked = (gene_ids.unsqueeze(2) != gene_target_ids.unsqueeze(1)).all(dim=2)
                idx_unmasked = (gene_target_ids.unsqueeze(2) == gene_ids.unsqueeze(1)).any(dim=2)
                idx_masked = (gene_target_ids.unsqueeze(2) != gene_ids.unsqueeze(1)).all(dim=2)
                print(idx_unmasked.shape, idx_masked.shape)
                idx_m = gene_target_ids[0, idx_masked[0]].cpu().numpy()
                idx_um = gene_target_ids[0, idx_unmasked[0]].cpu().numpy()
                inputs = gene_ids[0, :].cpu().numpy()
                m0 = set(idx_m).intersection(set(inputs))
                m1 = set(idx_um).intersection(set(inputs))
                print("MASK", len(m0), len(idx_m))
                print("UNMASK", len(m1), len(idx_um), idx_m)

            self.explained_var_masked(gene_pred[idx_masked], gene_targets[idx_masked].unsqueeze(-1))
            self.val_mse_masked(gene_pred[idx_masked], gene_targets[idx_masked].unsqueeze(-1))
            self.explained_var_unmasked(gene_pred[idx_unmasked], gene_targets[idx_unmasked].unsqueeze(-1))
            self.val_mse_unmasked(gene_pred[idx_unmasked], gene_targets[idx_unmasked].unsqueeze(-1))


        elif self.task_cfg["loss"] == "ZINB":

            # softplus_pi = F.softplus(-gene_pred[..., 2])  # log(1 + exp(-pi)) for stability
            # pi_prob = (1 - torch.sigmoid(gene_pred[..., 2]))
            pi_prob = (1 - torch.sigmoid(gene_pred[..., 1]))
            y = torch.exp(gene_pred[..., 0]) * pi_prob
            self.explained_var(torch.log(1 + y), gene_targets)
            self.val_mse(torch.log(1 + y), gene_targets)

            # self.ex_var_mse_val(gene_pred[..., 3], gene_targets)
            self.ex_var_mse_val(gene_pred[..., 2], gene_targets)
            self.log("exp_var_mse", self.ex_var_mse_val, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log("exp_var_masked", self.explained_var_masked, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("mse_masked", self.val_mse_masked, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("exp_var_unmasked", self.explained_var_unmasked, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("mse_unmasked", self.val_mse_unmasked, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        fn = f"{self.trainer.log_dir}/test_results.pkl"
        pickle.dump(self.results, open(fn, "wb"))
        self.results["epoch"] = self.current_epoch + 1
        for k in self.results_list:
            self.results[k] = []

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.task_cfg["learning_rate"],
            weight_decay=self.task_cfg["weight_decay"],
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.task_cfg["warmup_steps"],
            num_training_steps=self.task_cfg["total_steps"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }


class Annotation(pl.LightningModule):
    def __init__(
            self,
            network,
            task_cfg,
            **kwargs
    ):

        # Initialize superclass
        super().__init__()
        self.network = network

        for n, p in self.network.named_parameters():

            # if not ("feature" in n or "decoder" in n or "31" in n or "30" in n or "29" in n or "28" in n or "27" in n or "26" in n): # and not cond:
            if not "feature" in n:
                p.requires_grad_(False)

        for k, v in task_cfg.items():
            setattr(self, k, v)

        self.task_cfg = task_cfg
        self.cell_properties = task_cfg["cell_properties"]

        # Functions and metrics
        self._cell_properties_metrics()

        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None

        self.n_bins = task_cfg["n_bins"] if "n_bins" in task_cfg else None
        self.scale_target_depth = 3.0

        self._create_results_dict()
        self.source_code_copied = False

    def _cell_properties_metrics(self):

        self.cell_cross_ent = nn.ModuleDict()
        self.cell_mse = nn.ModuleDict()
        self.mse = nn.MSELoss()
        self.cell_accuracy = nn.ModuleDict()
        self.cell_explained_var = nn.ModuleDict()

        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                # discrete variable, set up cross entropy module
                # weight = torch.from_numpy(
                #    np.float32(np.clip(1 / cell_prop["freq"], 0.0001, 10000.0))
                # ) if self.balance_classes else None

                weight = torch.from_numpy(
                    np.float32(np.clip((np.max(cell_prop["freq"]) / cell_prop["freq"]), 1.0, 25.0))
                ) if self.balance_classes else None

                # print("Weight", weight)
                self.cell_cross_ent[k] = nn.CrossEntropyLoss(weight=weight, reduction="none", ignore_index=-100)
                # self.cell_cross_ent[k] = FocalLoss(len(cell_prop["values"]), gamma=2.0, alpha=2.0)
                self.cell_accuracy[k] = Accuracy(
                    task="multiclass", num_classes=len(cell_prop["values"]), average="macro",
                )
            else:
                # continuous variable, set up MSE module
                self.cell_mse[k] = nn.MSELoss(reduction="none")
                self.cell_explained_var[k] = ExplainedVariance()

    def _create_results_dict(self):

        self.results = {"epoch": 0, "gene_names": self.gene_names}
        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                n = len(cell_prop["values"])
                self.results[f"{k}_matrix"] = np.zeros((n, n))

    def _copy_source_code(self):

        target_dir = f"{self.trainer.log_dir}/code"
        os.mkdir(target_dir)
        base_dir = "/home/masse/work/perceiver/"
        src_files = [
            f"{base_dir}config.py",
            f"{base_dir}datasets.py",
            f"{base_dir}networks.py",
            f"{base_dir}tasks.py",
            f"{base_dir}train.py",
        ]
        for src in src_files:
            shutil.copyfile(src, f"{target_dir}/{os.path.basename(src)}")

    def _feature_scores(self, cell_pred, cell_targets, cell_mask):

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            idx = torch.nonzero(cell_mask[:, n])

            if cell_prop["discrete"]:
                pred_idx = torch.argmax(cell_pred[k], dim=-1).to(torch.int64)
                # pred_prob = F.softmax(cell_pred[k], dim=-1).to(torch.float32).detach().cpu().numpy()
                targets = cell_targets[:, n].to(torch.int64)
                self.cell_accuracy[k].update(pred_idx[idx, 0], targets[idx])

                idx1 = pred_idx[idx, 0].cpu().numpy()
                idx0 = targets[idx].cpu().numpy()
                for i, j in zip(idx0, idx1):
                    self.results[f"{k}_matrix"][i, j] += 1

            else:
                pred = cell_pred[k][:, 0]
                try:  # rare error
                    self.cell_explained_var[k].update(pred[idx], cell_targets[idx, n])
                except:
                    self.cell_explained_var[k].update(pred, cell_targets[idx, n])

        for k, v in self.cell_accuracy.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.cell_explained_var.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def _feature_loss(
            self,
            cell_pred: Dict[str, torch.Tensor],
            cell_prop_vals: torch.Tensor,
            cell_mask: torch.Tensor,
    ):

        cell_loss = 0.0
        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if cell_prop["discrete"]:
                loss = self.cell_cross_ent[k](torch.squeeze(cell_pred[k]), cell_prop_vals[:, n].to(torch.int64))
                cell_loss += (loss * cell_mask[:, n]).mean()
            else:
                loss = self.cell_mse[k](torch.squeeze(cell_pred[k]), cell_prop_vals[:, n])
                cell_loss += (loss * cell_mask[:, n]).mean()

        if self.training:
            self.log("cell_loss_train", cell_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("cell_loss_val", cell_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return cell_loss

    def training_step(self, batch, batch_idx):

        (
            gene_ids, gene_target_ids, gene_vals, gene_targets,
            depths, cell_prop_vals, cell_prop_mask, _
        ) = batch

        # depths[:, 0] = depths[:, 0] * self.scale_target_depth
        d = torch.exp(depths[:, 0])
        depths[:, 0] = torch.log(d * 1)

        gene_pred, _, feature_pred = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, depths=depths,
        )

        # print("TRAIN", gene_pred.mean())

        feature_loss = self._feature_loss(feature_pred, cell_prop_vals, cell_prop_mask)
        # gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        gene_loss = 0.0

        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("feature_loss", feature_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return feature_loss + 500 * gene_loss

    def validation_step(self, batch, batch_idx):

        (
            gene_ids, gene_target_ids, gene_vals, gene_targets,
            depths, cell_prop_vals, cell_prop_mask, _
        ) = batch

        gene_pred, _, feature_pred = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, depths=depths,
        )

        d = torch.exp(depths[:, 0])
        depths[:, 0] = torch.log(d * 1)

        self._feature_scores(feature_pred, cell_prop_vals, cell_prop_mask)
        self._feature_loss(feature_pred, cell_prop_vals, cell_prop_mask)

    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        fn = f"{self.trainer.log_dir}/test_results_ep{self.current_epoch}.pkl"

        pickle.dump(self.results, open(fn, "wb"))
        self.results["epoch"] = self.current_epoch + 1
        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                n = len(cell_prop["values"])
                self.results[f"{k}_matrix"] = np.zeros((n, n))

    def configure_optimizers(self):

        # params = [p for n, p in self.network.named_parameters() if not "gmlp" in n]
        out_params = [p for n, p in self.network.named_parameters() if "feature_decoder.cell_type.gene_mlp" in n]
        other_params = [p for n, p in self.network.named_parameters() if
                        "feature_decoder.cell_type.decoder_cross_attn" in n]

        return torch.optim.AdamW(
            # [
            #    {"params": out_params0, "lr": self.task_cfg["learning_rate"]},
            #    {"params": params1, "lr": self.task_cfg["learning_rate"]},
            # ],
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
        )


class MSELoss_atac(pl.LightningModule):
    def __init__(
            self,
            network,
            task_cfg,
            **kwargs
    ):

        # Initialize superclass
        super().__init__()
        self.network = network
        self.task_cfg = task_cfg

        self.cell_prop_cross_ent = None
        self.cell_prop_accuracy = None
        self.cell_prop_mse = None
        self.cell_prop_explained_var = None

        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None
        self.n_bins = task_cfg["n_bins"] if "n_bins" in task_cfg else None

        if self.task_cfg["predict_atac"]:
            self.cross_ent = nn.BCEWithLogitsLoss(weight=None)
            self.atac_accuracy = Accuracy(task='binary', average='macro', num_classes=2)
        else:
            self.mse = nn.MSELoss()
            self.gene_explained_var = ExplainedVariance()
            self.metrics = MetricCollection([ExplainedVariance()])

        self._create_results_dict()
        self.source_code_copied = False

    def _create_results_dict(self):

        self.results = {"epoch": 0, "gene_names": self.gene_names}
        # for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    self.results[k] = []
        # N = len(self.gene_names)
        N = 16594
        # self.results["target_gene_counts"] = np.zeros(N)
        # self.results["target_gene_preds"] = np.zeros(N)
        # self.results["atac_data"] = np.zeros((200, N, 128), dtype=np.float32)
        # self.results["pred_atac_data"] = np.zeros((200, N, 128), dtype=np.float32)
        # self.gcount = np.zeros(N)

    def _copy_source_code(self):

        target_dir = f"{self.trainer.log_dir}/code"
        os.mkdir(target_dir)
        base_dir = "/home/masse/work/perceiver/"
        src_files = [
            f"{base_dir}config_atac.py",
            f"{base_dir}datasets.py",
            f"{base_dir}networks.py",
            f"{base_dir}tasks.py",
            f"{base_dir}train_atac.py",
        ]
        for src in src_files:
            shutil.copyfile(src, f"{target_dir}/{os.path.basename(src)}")

    def training_step(self, batch, batch_idx):

        # atac_chr, atac_pos, atac_pos_abs, gene_ids, gene_vals, gene_target_ids, gene_target_vals, padding_mask_atac, padding_mask_genes = batch
        # gene_pred, latent = self.network.forward(
        #    atac_chr, atac_pos, atac_pos_abs, gene_ids, gene_vals, gene_target_ids, padding_mask_atac,
        #    padding_mask_genes,
        # )
        atac_gene_based, gene_ids, gene_vals, gene_target_ids, gene_target_vals, padding_mask_atac, padding_mask_genes = batch
        gene_pred, atac_pred, latent = self.network.forward(
            atac_gene_based, gene_ids, gene_vals, gene_target_ids, padding_mask_atac, padding_mask_genes,
        )

        if self.task_cfg["predict_atac"]:
            loss = self.cross_ent(atac_pred, atac_gene_based.squeeze())
        else:
            loss = self.mse(gene_pred, gene_target_vals.unsqueeze(2))

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # atac_chr, atac_pos, atac_pos_abs, gene_ids, gene_vals, gene_target_ids, gene_target_vals, padding_mask_atac, padding_mask_genes = batch
        # gene_pred, latent = self.network.forward(
        #    atac_chr, atac_pos, atac_pos_abs, gene_ids, gene_vals, gene_target_ids, padding_mask_atac,
        #    padding_mask_genes,
        # )
        atac_gene_based, gene_ids, gene_vals, gene_target_ids, gene_target_vals, padding_mask_atac, padding_mask_genes = batch
        gene_pred, atac_pred, latent = self.network.forward(
            atac_gene_based, gene_ids, gene_vals, gene_target_ids, padding_mask_atac, padding_mask_genes,
        )
        """
        for i in range(gene_pred.shape[0]):
            idx = gene_target_ids[i, :].detach().cpu().numpy()
            y = np.squeeze(gene_pred[i, :].to(torch.float32).detach().cpu().numpy())
            self.results["target_gene_counts"][idx] += 1
            self.results["target_gene_preds"][idx] += y
        """

        # self.results["gene_target_ids"].append(gene_target_ids.to(torch.int16).detach().cpu().numpy())
        # self.results["gene_vals"].append(gene_vals.detach().cpu().numpy())
        # self.results["gene_ids"].append(gene_ids.detach().cpu().numpy())
        # self.results["gene_targets"].append(gene_targets.detach().cpu().numpy())
        # self.results["gene_pred"].append(gene_pred.to(torch.float32).detach().cpu().numpy())
        # self.results["atac_data"].append(atac_gene_based.to(torch.float32).detach().cpu().numpy())
        # self.results["pred_atac_data"].append(atac_pred.to(torch.float32).detach().cpu().numpy())

        """
        x0 = atac_gene_based.to(torch.float32).detach().cpu().numpy()
        y0 = atac_pred.to(torch.float32).detach().cpu().numpy()

        for i in range(48):
            idx = gene_target_ids[i, :].detach().cpu().numpy()
            for k, j in enumerate(idx):
                if self.gcount[j] < 199:

                    #print(self.count[j], j)
                    self.results["atac_data"][int(self.gcount[j]), j, :] += x0[i, k, 0, :]
                    self.results["pred_atac_data"][int(self.gcount[j]), j, :] += y0[i, k]
                    self.gcount[j] += 1

        """

        # self.results["atac_data"] += atac_gene_based[gene_target_ids.to(torch.float32).detach().cpu().numpy()

        if self.task_cfg["predict_atac"]:
            # print("AAAA", atac_pred.shape, atac_gene_based.shape)
            self.atac_accuracy(atac_pred, atac_gene_based.squeeze())
            self.log("atac_acc", self.atac_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.gene_explained_var(gene_pred, gene_target_vals.unsqueeze(2))
            self.log("gene_ex", self.gene_explained_var, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        """
        if self.perturb_idx is None:
            fn = f"{self.trainer.log_dir}/test_results.pkl"
        else:
            fn = f"{self.trainer.log_dir}/test_results_ep{self.current_epoch}.pkl"
        #for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    if len(self.results[k]) > 0:
        #        self.results[k] = np.concatenate(self.results[k], axis=0)

        self.results["epoch"] = self.current_epoch + 1
        #for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    self.results[k] = []
        N = len(self.gene_names)
        #self.results["target_gene_counts"] = np.zeros(N)
        #self.results["target_gene_preds"] = np.zeros(N)
        """
        # self.results["atac_data"] = np.mean(np.stack(self.results["atac_data"]), axis=(0,1))
        # self.results["pred_atac_data"] = np.mean(np.stack(self.results["pred_atac_data"]), axis=(0, 1))
        fn = f"{self.trainer.log_dir}/test_results.pkl"
        # pickle.dump(self.results, open(fn, "wb"))
        N = 16594
        # self.results["target_gene_counts"] = np.zeros(N)
        # self.results["target_gene_preds"] = np.zeros(N)
        # self.results["atac_data"] = np.zeros((200, N, 128), dtype=np.float32)
        # self.results["pred_atac_data"] = np.zeros((200, N, 128), dtype=np.float32)
        # self.gcount = np.zeros(N)

    def configure_optimizers(self):

        base_params = [p for n, p in self.network.named_parameters() if "SubID" not in n]
        reversal_params = [p for n, p in self.network.named_parameters() if "SubID" in n]

        return torch.optim.AdamW(
            [
                {"params": base_params},
                {"params": reversal_params, "lr": self.task_cfg["learning_rate"] * 10, "weight_decay": 0.0},
            ],
            lr=self.task_cfg["learning_rate"],
            weight_decay=self.task_cfg["weight_decay"],
        )

    def optimizer_step(
            self,
            current_epoch,
            batch_nb,
            optimizer,
            closure,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
            min_lr=1e-8,
    ):

        # warm up lr
        if self.trainer.global_step < self.task_cfg["warmup_steps"]:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1) / float(self.task_cfg["warmup_steps"]),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.task_cfg["learning_rate"]


        elif self.trainer.global_step > self.task_cfg["decay_steps"]:
            lr_scale = self.task_cfg["decay"] ** (self.trainer.global_step - self.task_cfg["decay_steps"])
            lr_scale = max(min_lr, lr_scale)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.task_cfg["learning_rate"]

        # update params
        optimizer.step(closure=closure)


class AdverserialLoss(MSELoss):
    def __init__(
            self,
            network,
            task_cfg,
            adv_cell_prop: str = "SubID",
            adv_loss_ratio: float = 0.05,
            adv_threshold: float = 0.0,
            **kwargs
    ):
        # Initialize superclass
        super().__init__(network, task_cfg, automatic_optimization=False, **kwargs)
        print("ADV LOSS COEFF", adv_loss_ratio)
        self.network_params = [p for n, p in self.network.named_parameters() if adv_cell_prop not in n]
        self.adv_params = [p for n, p in self.network.named_parameters() if adv_cell_prop in n]
        self.adv_cell_prop = adv_cell_prop
        self.adv_loss_ratio = adv_loss_ratio
        self.adv_threshold = adv_threshold

        self.adv_accuracy = nn.ModuleDict()
        self.adv_accuracy[adv_cell_prop] = Accuracy(
            task="multiclass", num_classes=len(self.cell_properties[adv_cell_prop]["values"]), average="macro",
        )
        self.steps = 0.0

    def training_step(self, batch, batch_idx):

        opt0, opt1 = self.configure_optimizers()
        opt0, opt1 = self.update_lr(opt0, opt1)
        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets, cell_class_id = batch
        gene_pred, cell_prop_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, cell_prop_ids, gene_vals, cell_class_id, key_padding_mask,
        )

        gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        cell_prop_loss = 0.0
        adv_loss = 0.0

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if cell_prop["discrete"]:
                # discrete variable, use cross entropy
                # class values of -1 will be masked out
                idx = torch.where(cell_prop_targets[:, n] >= 0)[0]
                p_loss = self.cell_prop_cross_ent[k](
                    cell_prop_pred[k][idx], cell_prop_targets[idx, n].to(torch.int64)
                )

                if k == self.adv_cell_prop:
                    if p_loss < self.adv_threshold:
                        cell_prop_loss -= self.adv_loss_ratio * p_loss
                    opt1.zero_grad()
                    self.manual_backward(p_loss, retain_graph=True)
                    self.log(
                        "adv_acc", p_loss,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
                else:
                    cell_prop_loss += p_loss

            else:
                # continuous variable, use MSE
                # class values less than -999 or greater than 999 will be masked out
                idx = torch.where(cell_prop_targets[:, n] > -999)[0]
                cell_prop_loss += self.cell_prop_mse[k](
                    cell_prop_pred[k][idx], cell_prop_targets[idx, n]
                )

        loss = self.task_cfg["gene_weight"] * gene_loss + self.task_cfg["cell_prop_weight"] * cell_prop_loss

        opt0.zero_grad()
        self.manual_backward(loss)

        # clip gradients
        self.clip_gradients(opt0, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(opt1, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # apply gradient step
        opt0.step()
        opt1.step()

        # log results
        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "cell_loss",
            cell_prop_loss + self.adv_loss_ratio * adv_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "adv_acc", adv_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_train_epoch_end(self):
        """Need to manually save checkpoints as automatic checkpointing is disabled for some reason..."""
        ckpt_path = self.logger.log_dir + "/saved_model.ckpt"
        self.trainer.save_checkpoint(ckpt_path)

    def update_lr(self, opt0, opt1):

        self.steps += 1.0
        lr = self.task_cfg["learning_rate"] * np.minimum(
            self.steps ** (-0.5), self.steps * float(self.task_cfg["warmup_steps"]) ** (-1.5))
        for opt in [opt0, opt1]:
            for pg in opt.param_groups:
                pg["lr"] = lr
        return opt0, opt1

    def configure_optimizers(self):

        opt0 = torch.optim.AdamW(
            self.network_params,
            lr=0.0,
            weight_decay=self.task_cfg["weight_decay"],
        )
        opt1 = torch.optim.AdamW(
            self.adv_params,
            lr=0.0,
            weight_decay=self.task_cfg["weight_decay"],
        )

        return opt0, opt1
