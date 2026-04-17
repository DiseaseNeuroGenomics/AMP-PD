from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False,
    ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def _convert_target(self, target):

        target = torch.clip(target, 0, self.num_classes - 1)

        if self.num_classes == 2:
            new_target = (
                0.95 * F.one_hot(target, num_classes=self.num_classes) +
                0.05 * F.one_hot(1 - target, num_classes=self.num_classes)
            )
        else:
            new_target = 0.9 * F.one_hot(target, num_classes=self.num_classes)
            t0 = torch.clip(target - 1, 0, self.num_classes - 1)
            t1 = torch.clip(target + 1, 0, self.num_classes - 1)
            new_target += 0.05 * F.one_hot(t0, num_classes=self.num_classes)
            new_target += 0.05 * F.one_hot(t1, num_classes=self.num_classes)

        return new_target

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                # labels = F.one_hot(labels, num_classes=self.num_classes)
                labels = self._convert_target(labels)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        poly1 = poly1.sum(dim=-1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.alpha = alpha
        print(f"FOCAL LOSS, num classes {num_classes}")

    def _convert_target(self, target):

        target = torch.clip(target, 0, self.num_classes - 1)

        if self.num_classes == 2:
            new_target = (
                0.95 * F.one_hot(target, num_classes=self.num_classes) +
                0.05 * F.one_hot(1 - target, num_classes=self.num_classes)
            )
        else:
            new_target = 0.9 * F.one_hot(target, num_classes=self.num_classes)
            t0 = torch.clip(target - 1, 0, self.num_classes - 1)
            t1 = torch.clip(target + 1, 0, self.num_classes - 1)
            new_target += 0.05 * F.one_hot(t0, num_classes=self.num_classes)
            new_target += 0.05 * F.one_hot(t1, num_classes=self.num_classes)

        return new_target


    def forward(self, input, target):

        new_target = self._convert_target(target)
        # new_target = F.one_hot(target, num_classes=self.num_classes)
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        loss = - new_target * log_prob * (1 - prob) ** self.gamma
        return self.alpha * loss.sum(dim=-1)


class GroupDRO:

    def __init__(
        self,
        groups: Optional[List] = None,
        balances: Optional[List] = None,
        alpha: float = 0.99,
    ):

        if groups is None:
            self.groups = [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 998, 999]
        else:
            self.groups = groups
        self.alpha = alpha
        self.batch_loss = {g: 0.0 for g in self.groups}
        self.batch_count = {g: 0 for g in self.groups}
        self.running_loss = {g: 0.0 for g in self.groups}
        self.weights = {g: 1.0 for g in self.groups}

        if balances is None:
            self.balances = [
                [0, 1, 60, 61],
                [10, 11, 50, 51],
                [20, 21, 40, 41],
                [30, 31],
            ]
        else:
            self.balances = balances

    def update_batch(self, loss, mask, group_idx):

        for l, m, g in zip(loss.tolist(), mask.tolist(), group_idx):
            self.batch_loss[g] += l * m
            self.batch_count[g] += m


    def update_weights(self):

        for g in self.groups:
            if self.batch_count[g] > 1:
                self.running_loss[g] = (
                    self.alpha * self.running_loss[g] + (1 - self.alpha) * self.batch_loss[g] / self.batch_count[g]
                )

            # reset
            self.batch_loss[g] = 0.0
            self.batch_count[g] = 0

        for balances in self.balances:
            mean_weights = np.mean(np.array([self.running_loss[g] for g in balances]))
            for g in balances:
                self.weights[g] = self.running_loss[g] / mean_weights


    def get_weights(self, group_idx):

        weights = [self.weights[g] for g in group_idx]
        return torch.from_numpy(np.array(weights))
