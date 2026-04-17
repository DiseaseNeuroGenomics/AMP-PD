from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # new_target = self._convert_target(target)
        new_target = F.one_hot(target, num_classes=self.num_classes)
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        loss = - new_target * log_prob * (1 - prob) ** self.gamma
        return self.alpha * loss.sum(dim=-1)


class ZINB(nn.Module):
    """
    Zero-Inflated Negative Binomial (ZINB) loss.

    Args:
        x: observed gene expression (batch_size, n_genes)
        mu: predicted mean (batch_size, n_genes)
        theta: predicted dispersion (batch_size, n_genes)
        pi: predicted zero-inflation logits (batch_size, n_genes)  (logits! before sigmoid)
        eps: small number for numerical stability
        reduction: 'mean' or 'sum'

    Returns:
        Loss value
    """
    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, mu, theta, pi, eps=1e-8, reduction='mean'):

        """
        Compute the ZINB (Zero-Inflated Negative Binomial) loss.

        Parameters:
        -----------
        x : torch.Tensor
            Observed count data (batch_size x genes).
        mu : torch.Tensor
            Mean of Negative Binomial (after softplus or exp to ensure positivity).
        theta : torch.Tensor
            Inverse dispersion (after softplus to ensure positivity).
        pi : torch.Tensor
            Dropout logits (before sigmoid).
        eps : float
            Small number for numerical stability.
        reduction : str
            'mean' | 'sum' | 'none' : how to reduce the final loss.

        Returns:
        --------
        loss : torch.Tensor
            Scalar or vector loss.
        """
        dtype = x.dtype

        theta = torch.clamp(theta, min=eps)
        mu = torch.clamp(mu, min=eps)

        #  uses log(sigmoid(x)) = -softplus(-x)
        softplus_pi = F.softplus(-pi)
        # eps to make it positive support and taking the log
        # print("theta", theta.shape, "mu", mu.shape)
        log_theta_mu_eps = torch.log(theta + mu + eps)
        pi_theta_log = -pi + theta * (torch.log(theta + eps) - log_theta_mu_eps)

        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < eps).to(dtype), case_zero)

        case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        mul_case_non_zero = torch.mul((x > eps).to(dtype), case_non_zero)

        loss = mul_case_zero + mul_case_non_zero

        theta_reg = torch.mean((theta - 1.0) ** 2)
        pi_prob = torch.sigmoid(pi)
        pi_reg = torch.mean(-(0.5 * torch.log(pi_prob + eps) + 0.5 * torch.log(1 - pi_prob + eps)))

        regulairization = 1e-4 * theta_reg + 1e-4 * pi_reg

        loss = loss - regulairization


        if reduction == 'mean':
            return -loss.mean()
        elif reduction == 'sum':
            return -loss.sum()
        elif reduction == 'none':
            return -loss  # 'none'