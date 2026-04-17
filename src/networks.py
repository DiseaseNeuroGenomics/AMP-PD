from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from modules import CellDecoder,  FCLayers

class FeedforwardNetwork(nn.Module):

    """Simple MLP style network"""

    def __init__(
        self,
        n_input: int,
        n_layers: int = 1,
        cell_properties: Optional[Dict[str, Any]] = None,
        batch_properties: Optional[Dict[str, Any]] = None, # not used in paper
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        input_dropout_rate: float = 0.0,
        grad_reverse_dict: Optional[Dict] = None,
        cell_decoder_hidden_layer: bool = False,
        layer_norm: bool = False,
        log_first: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.log_first = log_first
        self.dropout = nn.Dropout(p=float(input_dropout_rate))
        self.n_layers = n_layers
        if n_layers > 0:
            self.encoder = FCLayers(
                n_in=n_input,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
                batch_properties=batch_properties,
            )


        self.cell_decoder = CellDecoder(
            latent_dim=n_hidden if n_layers > 0 else n_input,
            cell_properties=cell_properties,
            grad_reverse_dict=grad_reverse_dict,
            use_hidden_layer=cell_decoder_hidden_layer,
        )

        self.drop = nn.Dropout(p=float(input_dropout_rate)) if input_dropout_rate > 0 else nn.Identity()


    def forward(
        self,
        x: torch.Tensor,
        batch_labels: torch.Tensor,
        batch_mask: torch.Tensor,
    ):

        if self.log_first:
            x = torch.log(1.0 + x)

        x = self.drop(x)

        if self.n_layers > 0:
            x = self.encoder(x, batch_labels, batch_mask)

        return self.cell_decoder(x)


def load_model(model_save_path, model):

    params_loaded = []
    non_network_params = []
    state_dict = {}
    ckpt = torch.load(model_save_path)
    key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
    for k, v in ckpt[key].items():
        print()
        print(f"From saved model: {k}")
        if "cell_property" in k:
            non_network_params.append(k)
        elif "network" in k:
            k = k.split(".")
            k = ".".join(k[1:])

        for n, p in model.named_parameters():
            if n == k:
                #pass
                print(k, p.size(), v.size(), p.size() == v.size())
            if n == k and p.size() == v.size():
                state_dict[k] = v
                params_loaded.append(n)

    model.load_state_dict(state_dict, strict=True)
    print(f"Number of params loaded: {len(params_loaded)}")
    print(f"Non-network parameters not loaded: {non_network_params}")
    return model