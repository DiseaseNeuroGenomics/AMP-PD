from typing import Any, Dict, Optional
import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np


class FCLayers(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.0,
        batch_properties: Optional[Dict[str, Any]] = None,
        layer_norm: bool = True,
    ):
        super().__init__()

        # we will inject the batch variables into the first hidden layer
        self.batch_dims = [] if batch_properties is None else [
            len(batch_properties[k]["values"]) for k in batch_properties.keys()
        ]
        total_batch_dims = int(np.sum(self.batch_dims))

        input_dim = [n_in + total_batch_dims] + (n_layers - 1) * [n_hidden + total_batch_dims]
        output_dim = n_layers * [n_hidden]

        self.fc_layers = nn.ModuleList([])
        for n in range(n_layers):
            layers = []
            linear = nn.Linear(input_dim[n], output_dim[n])
            layers.append(linear)

            layers.append(nn.ReLU())
            if layer_norm:
                layers.append(nn.LayerNorm(output_dim[n], elementwise_affine=True))
            layers.append(nn.Dropout(p=float(dropout_rate)))
            self.fc_layers.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor, batch_vals: torch.Tensor, batch_mask: torch.Tensor):

        for layer in self.fc_layers:
            if batch_vals is not None: # batch_vals not used in paper
                batch_vals[batch_vals < 0] = 0
                batch_vars = []
                for n in range(len(self.batch_dims)):
                    b = F.one_hot(batch_vals[:, n], num_classes=self.batch_dims[n])
                    b = 10.0 * b * batch_mask[:, n: n + 1]
                    batch_vars.append(b)
                x = torch.cat((x, *batch_vars), dim=-1)


            x = layer(x)

        return x


class CellDecoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        cell_properties: Dict[str, Any],
        hidden_dim: int = 256,
        batch_properties: Optional[Dict[str, Any]] = None,
        grad_reverse_dict: Optional[Dict] = None,
        use_hidden_layer: bool = True,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.batch_dims = 0 if batch_properties is None else [
            len(batch_properties[k]["values"]) for k in batch_properties.keys()
        ]
        self.latent_dim = latent_dim + np.sum(self.batch_dims)
        print("Batch and latent dims", np.sum(self.batch_dims) , self.latent_dim)

        self.cell_properties = cell_properties
        self.cell_out = nn.ModuleDict()

        self.grad_reverse_dict = grad_reverse_dict
        if grad_reverse_dict is not None:
            self.grad_reverse = nn.ModuleDict()
            for k, v in grad_reverse_dict.items():
                print(f"Grad reverse {k}: {v}")
                self.grad_reverse[k] = GradReverseLayer(v)

        for k, cell_prop in cell_properties.items():
            # the output size of the cell property prediction MLP will be 1 if the property is continuous;
            # if it is discrete, then it will be the length of the possible values
            n_targets = 1 if not cell_prop["discrete"] else len(cell_prop["values"])
            if use_hidden_layer:
                print("Cell decoder hidden layer set to TRUE")
                self.cell_out[k] = nn.Sequential(
                    nn.Linear(self.latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=float(dropout_rate)),
                    nn.Linear(hidden_dim, n_targets),
                )
            else:
                print("Cell decoder hidden layer set to FALSE")
                self.cell_out[k] = nn.Linear(self.latent_dim, n_targets, bias=True)


    def forward(
        self,
        latent: torch.Tensor,
        batch_vals: Optional[torch.Tensor] = None,
        batch_mask: Optional[torch.Tensor] = None,
    ):

        # Predict cell properties
        if np.sum(self.batch_dims) > 0 and batch_vals is not None:
            batch_vals[batch_vals < 0] = 0
            batch_vars = []
            for n in range(len(self.batch_dims)):
                b = F.one_hot(batch_vals[:, n], num_classes=self.batch_dims[n])
                b = b * batch_mask[:, n: n+1]
                batch_vars.append(b)

            latent = torch.cat((latent, *batch_vars), dim=-1)

        output = {}
        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
            if self.grad_reverse_dict is not None and k in self.grad_reverse_dict.keys():
                x = self.grad_reverse[k](latent)
            else:
                x = latent.detach() if cell_prop["stop_grad"] else latent

            output[k] = self.cell_out[k](x)

        return output

