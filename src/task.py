import os
from typing import Any, Dict, List, Optional

import shutil
import pickle
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.distributions import Normal, kl_divergence as kl
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torchmetrics import MetricCollection, ExplainedVariance
from torchmetrics.classification import Accuracy

from losses import FocalLoss

import scipy.stats as stats

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class StandardLoss(pl.LightningModule):
    def __init__(
        self,
        network,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.0005,
        warmup_steps: float = 2000.0,
        weight_decay: float = 0.1,
        l1_lambda: float = 0.0,
        save_gene_vals: bool = False,
        balance_classes: bool = False,
        focal_loss: bool = False,
        focal_loss_gamma: float = 2.0,
        adam_betas: List[float] = [0.9, 0.998],
        adam_eps: float = 1e-7,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.network = network
        self.cell_properties = cell_properties
        self.batch_properties = batch_properties
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.save_gene_vals = save_gene_vals
        self.balance_classes = balance_classes
        self.focal_loss = focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.label_smoothing = label_smoothing

        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay: {self.weight_decay}")

        self._cell_properties_metrics()
        self._create_results_dict()
        self.train_step = 0
        self.source_code_copied = False


    def _create_results_dict(self):

        self.results = {"epoch": 0}
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results[f"pred_{k}"] = []

        if self.batch_properties is not None:
            for k in self.batch_properties.keys():
                self.results[k] = []
        self.results["cell_mask"] = []
        self.results["px_r"] = []
        self.results["cell_idx"] = []

    def _cell_properties_metrics(self):

        self.cell_cross_ent = nn.ModuleDict()
        self.cell_mse = nn.ModuleDict()
        self.cell_accuracy = nn.ModuleDict()
        self.cell_explained_var = nn.ModuleDict()

        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                # discrete variable, set up cross entropy module
                weight = torch.from_numpy(
                    np.float32(np.clip(1 / cell_prop["freq"], 0.1, 10.0))
                ) if self.balance_classes else None
                if self.focal_loss:
                    self.cell_cross_ent[k] = FocalLoss(len(cell_prop["values"]), gamma=self.focal_loss_gamma, alpha=2.0)
                else:
                    self.cell_cross_ent[k] = nn.CrossEntropyLoss(
                        weight=weight,
                        reduction="none",
                        ignore_index=-100,
                        label_smoothing=self.label_smoothing,
                    )
                # self.cell_cross_ent[k] = Poly1FocalLoss(len(cell_prop["values"]), gamma=2.0)
                self.cell_accuracy[k] = Accuracy(
                    task="multiclass", num_classes=len(cell_prop["values"]), average="macro",
                )
            else:
                # continuous variable, set up MSE module
                self.cell_mse[k] = nn.MSELoss(reduction="none")
                self.cell_explained_var[k] = ExplainedVariance()

    def copy_source_code(self, version_num):

        target_dir = f"{self.trainer.log_dir}/code"
        os.mkdir(target_dir)
        src_files = [
            "config.py",
            "data.py",
            "modules.py",
            "networks.py",
            "task.py",
            "train.py",
            "losses.py",
        ]
        for src in src_files:
            shutil.copyfile(src, f"{target_dir}/{os.path.basename(src)}")
        self.source_code_copied = True

    def validation_step(self, batch, batch_idx):

        gene_vals, cell_targets, cell_mask, batch_labels, batch_mask, cell_idx, _, _ = batch

        cell_pred = self.network(gene_vals, batch_labels, batch_mask)

        self.results["cell_idx"].append(cell_idx.detach().cpu().numpy())

        self.cell_scores(cell_pred, cell_targets, cell_mask)

        loss = self._cell_loss(cell_pred, cell_targets, cell_mask, batch_labels)

        if self.batch_properties is not None:
            for n, k in enumerate(self.batch_properties.keys()):
                self.results[k].append(batch_labels[:, n].detach().cpu().numpy())


        return loss


    def on_validation_epoch_end(self):

        for k in self.cell_properties.keys():
            if len(self.results[k]) > 0:
                self.results[k] = np.concatenate(self.results[k], axis=0)
                if k != "SubID":
                    self.results[f"pred_{k}"] = np.concatenate(self.results[f"pred_{k}"], axis=0)


        if self.batch_properties is not None:
            for k in self.batch_properties.keys():
                self.results[k] = np.concatenate(self.results[k], axis=0)
        if self.save_gene_vals:
            self.results["px_r"] = np.concatenate(self.results["px_r"], axis=0)
        self.results["cell_mask"] = np.concatenate(self.results["cell_mask"], axis=0)
        self.results["cell_idx"] = np.concatenate(self.results["cell_idx"], axis=0)

        self.results["SubID"] = None

        v = self.trainer.logger.version
        if not self.source_code_copied:
            self.copy_source_code(v)


        fn = f"{self.trainer.log_dir}/test_results_ep{self.current_epoch}.pkl"
        if self.current_epoch > -1:
            pickle.dump(self.results, open(fn, "wb"))

        self.results["epoch"] = self.current_epoch + 1
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results[f"pred_{k}"] = []

        if self.batch_properties is not None:
            for k in self.batch_properties.keys():
                self.results[k] = []
        self.results["cell_mask"] = []
        self.results["px_r"] = []
        self.results["cell_idx"] = []

    def cell_scores(self, cell_pred, cell_targets, cell_mask, brain_region=None):


        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            idx = torch.nonzero(cell_mask[:, n])

            if cell_prop["discrete"]:
                pred_idx = torch.argmax(cell_pred[k], dim=-1).to(torch.int64)
                pred_prob = F.softmax(cell_pred[k], dim=-1).to(torch.float32).detach().cpu().numpy()

                targets = cell_targets[:, n].to(torch.int64)

                self.cell_accuracy[k].update(pred_idx[idx][None, :], targets[idx][None, :])
                if k != "SubID":
                    self.results[k].append(targets.detach().cpu().numpy())
                    self.results["pred_" + k].append(pred_prob)
            else:

                pred = cell_pred[k][:, 0]

                try: # rare error
                    self.cell_explained_var[k].update(pred[idx], cell_targets[idx, n])
                except:
                    self.cell_explained_var[k].update(pred, cell_targets[idx, n])

                self.results[k].append(cell_targets[:, n].detach().cpu().numpy())
                self.results[f"pred_{k}"].append(pred.detach().cpu().to(torch.float32).numpy())


        self.results["cell_mask"].append(cell_mask.detach().cpu().to(torch.float32).numpy())
        #self.results["SubID"].append(subject)

        for k, v in self.cell_accuracy.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.cell_explained_var.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)


    def training_step(self, batch, batch_idx):

        self.train_step += 1

        gene_vals, cell_prop_vals, cell_mask, batch_labels, batch_mask, _, group_idx, _ = batch

        cell_pred  = self.network(gene_vals, batch_labels, batch_mask)

        loss = self._cell_loss(cell_pred, cell_prop_vals, cell_mask, batch_labels)

        return loss


    def _cell_loss(
        self,
        cell_pred: Dict[str, torch.Tensor],
        cell_prop_vals: torch.Tensor,
        cell_mask: torch.Tensor,
        batch_labels: torch.Tensor,
        opt_idx: Optional[int] = None,
    ):

        cell_loss = 0.0

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if cell_prop["discrete"]:
                loss = self.cell_cross_ent[k](cell_pred[k], cell_prop_vals[:, n].to(torch.int64))
                masked_loss = (loss * cell_mask[:, n]).mean()
                cell_loss += masked_loss

            else:
                alpha = 1 / self.cell_properties[k]["std"]**2
                loss = alpha * self.cell_mse[k](torch.squeeze(cell_pred[k]), cell_prop_vals[:, n])
                masked_loss = (loss * cell_mask[:, n]).mean()
                cell_loss += masked_loss

            if self.training:
                self.log(f"loss_{k}", masked_loss, on_step=False, on_epoch=True, prog_bar=False,
                         sync_dist=True)

        if self.training:
            self.log("cell_loss_train", cell_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("cell_loss_val", cell_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        return cell_loss


    def configure_optimizers(self):


        opt = torch.optim.SGD(
            self.network.parameters(),
            lr=self.learning_rate,
            momentum=self.adam_betas[0],
            weight_decay=self.weight_decay,
        )
            

        lr_scheduler = {
            'scheduler': WarmupSchedule(opt, warmup_steps=self.warmup_steps),
            'interval': 'step',
        }

        return {
            'optimizer': opt,
            'lr_scheduler': lr_scheduler,  # Changed scheduler to lr_scheduler
            'interval': 'step',
        }


class WarmupConstantAndDecaySchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.alpha = np.sqrt(warmup_steps)
        super(WarmupConstantAndDecaySchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        arg1 = (step + 1) ** (-0.5)
        arg2 = (step + 1) * (self.warmup_steps ** -1.5)
        return self.alpha * np.minimum(arg1, arg2)

class WarmupSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return np.clip(step / self.warmup_steps, 0.0, 1.0)








