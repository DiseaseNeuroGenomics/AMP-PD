from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.fft
from torch.distributions import Normal

import scipy.stats as stats

class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MLPEmbedding(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            n_input: int = 1,
            linear: bool = False,
            n_hidden: int = 32,
    ):
        super().__init__()
        print("Creating gene value embedding")

        self.n_input = n_input
        if linear:
            self.mlp = nn.Linear(n_input, embedding_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_input, n_hidden),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(n_hidden, embedding_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x = x.unsqueeze(-1) if self.n_input == 1 else x
        return self.mlp(x)


class SimpleHyenaBlock(nn.Module):
    def __init__(self, dim, kernel_size=64, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim or 4 * dim

        self.modulator = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )

        # One kernel per channel
        self.kernel = nn.Parameter(torch.randn(dim, kernel_size))

        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        B, L, D = x.shape
        x_proj = self.in_proj(x)
        mod = self.modulator(x)

        modulated = x_proj * mod  # elementwise modulation

        # FFT-based convolution per channel
        x_fft = torch.fft.rfft(modulated.transpose(1, 2), n=L)
        k_fft = torch.fft.rfft(self.kernel, n=L)
        y_fft = x_fft * k_fft.unsqueeze(0)  # broadcasting kernel
        y = torch.fft.irfft(y_fft, n=L).transpose(1, 2)

        return self.out_proj(y)


class CrossAttn(nn.Module):
    def __init__(
            self,
            query_dim: int,
            key_val_dim: int,
            num_heads: int = 4,
            dropout: float = 0.0,
            residual: bool = True,
            moe: bool = False,
    ):
        super().__init__()
        self.layernorm_kv = nn.LayerNorm(key_val_dim)
        self.layernorm_q = nn.LayerNorm(query_dim)
        self.residual = residual
        print("CrossAttn drop", dropout)
        self.cross_attn = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=0.0,
            kdim=key_val_dim,
            vdim=key_val_dim,
            batch_first=True,
        )

        if moe:
            self.out = MoELayer(query_dim, query_dim, num_experts=4, k=1, noisy_gating=True)
        else:
            self.out = MLP(query_dim, 4 * query_dim, dropout=dropout)

    def forward(
            self,
            query: torch.Tensor,
            key_val: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_k = self.layernorm_kv(key_val)
        norm_v = self.layernorm_kv(key_val)
        norm_q = self.layernorm_q(query)

        # TODO: do I need to output weights??
        latent, weights = self.cross_attn(norm_q, norm_k, norm_v, key_padding_mask)
        # residual connection
        if self.residual:
            latent = latent + query
        latent = self.out(latent) + latent

        return latent


class StackedFlexibleTransformer(nn.Module):
    def __init__(
            self,
            num_layers: int,
            query_input_dim: int,
            kv_input_dim: int,
            qk_dim: int,
            v_dim: int,
            output_dim: int,
            num_heads: int,
            mlp_ratio: int = 4,
            use_xformers: bool = True,
            dropout: float = 0.0,
            norm_fn: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            FlexibleAttentionBlock(
                query_input_dim=query_input_dim,
                kv_input_dim=kv_input_dim,
                qk_dim=qk_dim,
                v_dim=v_dim,
                output_dim=output_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_xformers=use_xformers,
                dropout=dropout,
                norm_fn=norm_fn,
            )
            for _ in range(num_layers)
        ])

    def forward(
            self,
            query: torch.Tensor,  # (B, L_q, D)
            key_value: Optional[torch.Tensor] = None,  # for cross-attn layers
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = query

        for block in self.blocks:
            x = block(x, x, key_padding_mask)

        return x


class FlexibleAttentionBlock(nn.Module):
    def __init__(
            self,
            query_input_dim: int,
            kv_input_dim: int,
            qk_dim: int,
            v_dim: int,
            output_dim: int,
            num_heads: int,
            mlp_ratio: int = 4,
            use_xformers: bool = True,
            dropout: float = 0.0,
            norm_fn: nn.Module = nn.LayerNorm,  # swap in RMSNorm if desired
            residual: bool = True,
    ):
        super().__init__()

        self.norm_q = norm_fn(query_input_dim)
        self.norm_kv = norm_fn(kv_input_dim)
        self.residual = residual

        self.attn = FlexibleAttention(
            query_input_dim=query_input_dim,
            kv_input_dim=kv_input_dim,
            qk_dim=qk_dim,
            v_dim=v_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            use_xformers=use_xformers,
            dropout=dropout,
        )

        hidden_dim = output_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        if query_input_dim != output_dim:
            self.project_q = nn.Linear(query_input_dim, output_dim)
        else:
            self.project_q = nn.Identity()

    def forward(
            self,
            query: torch.Tensor,  # (B, L_q, D_q)
            key_value: torch.Tensor,  # (B, L_kv, D_kv)
            key_padding_mask=None  # (B, L_kv)
    ) -> torch.Tensor:
        # Norm first (PreNorm transformer style)
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)

        # Attention + residual
        x = self.attn(q_norm, kv_norm, key_padding_mask=key_padding_mask)
        if self.residual:
            x = self.project_q(query) + x

        # MLP + residual
        x = x + self.mlp(x)
        return x


class FlexibleAttention(nn.Module):
    def __init__(
            self,
            query_input_dim: int,
            kv_input_dim: int,
            qk_dim: int,
            v_dim: int,
            output_dim: int,
            num_heads: int,
            use_xformers: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()
        assert qk_dim % num_heads == 0, "qk_dim must be divisible by num_heads"
        assert v_dim % num_heads == 0, "v_dim must be divisible by num_heads"

        self.use_xformers = use_xformers and HAS_XFORMERS
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.head_dim_qk = qk_dim // num_heads
        self.head_dim_v = v_dim // num_heads

        # Q/K/V projections
        self.q_proj = nn.Linear(query_input_dim, qk_dim)
        self.k_proj = nn.Linear(kv_input_dim, qk_dim)
        self.v_proj = nn.Linear(kv_input_dim, v_dim)

        torch.nn.init.orthogonal_(self.q_proj.weight)
        self.q_proj.weight.data *= 0.05
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.weight.data *= 0.05
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.weight.data *= 0.10

        # Output projection

        self.out_proj = nn.Linear(v_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        # (B, L, D) → (B, H, L, D_head)
        B, L, D = x.shape
        H = D // head_dim
        return x.view(B, L, H, head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, D_head) → (B, L, H * D_head)
        return x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), -1)

    def forward(
            self,
            query: torch.Tensor,  # (B, L_q, query_input_dim)
            key_value: torch.Tensor,  # (B, L_kv, kv_input_dim)
            key_padding_mask: Optional[torch.Tensor] = None  # (B, L_kv)
    ) -> torch.Tensor:
        B, L_q, _ = query.shape
        L_kv = key_value.size(1)

        Q = self._split_heads(self.q_proj(query), self.head_dim_qk)  # (B, H, L_q, Dq)
        K = self._split_heads(self.k_proj(key_value), self.head_dim_qk)  # (B, H, L_kv, Dq)
        V = self._split_heads(self.v_proj(key_value), self.head_dim_v)  # (B, H, L_kv, Dv)

        if self.use_xformers:
            attn_bias = None
            if key_padding_mask is not None:
                attn_bias = key_padding_mask[:, None, None, :]
                attn_bias = attn_bias.expand(B, self.num_heads, L_q, L_kv)
                attn_bias = torch.where(
                    attn_bias.to(torch.bool),  # shape broadcast
                    torch.tensor(float('-inf'), device=key_padding_mask.device, dtype=Q.dtype),
                    torch.tensor(0.0, device=key_padding_mask.device, dtype=Q.dtype)
                )

            attn_out = memory_efficient_attention(Q, K, V, attn_bias=attn_bias)
        else:
            # Manual attention fallback
            """
            Q_ = Q.transpose(1, 2).reshape(B * self.num_heads, L_q, self.head_dim_qk)
            K_ = K.transpose(1, 2).reshape(B * self.num_heads, L_kv, self.head_dim_qk)
            V_ = V.transpose(1, 2).reshape(B * self.num_heads, L_kv, self.head_dim_v)

            scores = torch.matmul(Q_, K_.transpose(-1, -2)) / self.head_dim_qk**0.5

            if key_padding_mask is not None:
                mask = key_padding_mask[:, None, :].expand(B, self.num_heads, L_kv).to(torch.bool)
                mask = mask.reshape(B * self.num_heads, 1, L_kv)
                scores.masked_fill_(mask, float('-inf'))

            weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(self.dropout(weights), V_)  # (B*H, L_q, Dv)
            attn_out = attn_out.view(B, self.num_heads, L_q, self.head_dim_v)
            """
            attn_mask = None
            if key_padding_mask is not None:
                # key_padding_mask: (B, L_k) → (B, 1, 1, L_k) → broadcast to (B, H, L_q, L_k)
                attn_mask = key_padding_mask[:, None, None, :].expand(Q.shape[0], self.num_heads, Q.shape[2],
                                                                      K.shape[2])
                attn_mask = attn_mask.to(Q.device)

            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False
            )

        out = self._merge_heads(attn_out)  # (B, L_q, v_dim)
        return self.out_proj(out)


class XformersMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, v_proj_dim=None, kq_proj_dim=None):
        super().__init__()

        v_proj_dim = embed_dim if v_proj_dim is None else v_proj_dim
        kq_proj_dim = embed_dim if kq_proj_dim is None else kq_proj_dim

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.v_head_dim = v_proj_dim // num_heads
        self.kq_head_dim = kq_proj_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, kq_proj_dim)
        self.k_proj = nn.Linear(embed_dim, kq_proj_dim)
        self.v_proj = nn.Linear(embed_dim, v_proj_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, query, key_value, key_padding_mask=None):
        """
        query: (B, L_q, query_dim)
        key_value: (B, L_kv, kv_dim)
        key_padding_mask: (B, L_kv) -> bool, True = pad
        """
        B, L_q, _ = query.shape
        L_kv = key_value.shape[1]

        # Linear projections
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        # Reshape to (B, L, H, D)
        Q = Q.view(B, L_q, self.num_heads, self.kq_proj_dim)
        K = K.view(B, L_kv, self.num_heads, self.kq_proj_dim)
        V = V.view(B, L_kv, self.num_heads, self.v_proj_dim)

        # Optional attention bias for masking
        attn_bias = None
        if key_padding_mask is not None:
            # Create additive mask: (B, L_kv, 1)
            attn_bias = key_padding_mask[:, None, None, :]
            attn_bias = attn_bias.expand(B, self.num_heads, L_q, L_kv)
            attn_bias = torch.where(
                attn_bias.to(torch.bool),  # shape broadcast
                torch.tensor(-1e9, device=key_padding_mask.device, dtype=Q.dtype),
                torch.tensor(0.0, device=key_padding_mask.device, dtype=Q.dtype)
            )

        # Apply efficient attention
        out = memory_efficient_attention(Q, K, V, attn_bias=attn_bias, p=self.dropout if self.training else 0.0)

        # Back to (B, L_q, H, D) → (B, L_q, query_dim)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)

        return self.out_proj(out)


class EncoderDeocderStack(nn.Module):

    def __init__(self, query_dim: int, gene_emb_dim: int, n_layers: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderDeocder(query_dim, gene_emb_dim, num_heads=num_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(
            self,
            latent_query: torch.Tensor,
            gene_query: torch.Tensor,
            key_val: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for n, layer in enumerate(self.layers):
            mask = key_padding_mask if n == 0 else None
            key_val, _ = layer(latent_query, gene_query, key_val, mask)

        return key_val


class EncoderDeocder(nn.Module):

    def __init__(self, query_dim: int, gene_emb_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layernorm_kv = nn.RMSNorm(gene_emb_dim)
        self.layernorm_q = nn.RMSNorm(query_dim)

        self.layernorm_g = nn.RMSNorm(gene_emb_dim)
        self.layernorm_l = nn.RMSNorm(query_dim)

        """
        self.encoder = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=0.0,
            kdim=gene_emb_dim,
            vdim=gene_emb_dim,
            batch_first=True,
        )

        self.decoder= nn.MultiheadAttention(
            gene_emb_dim,
            num_heads,
            dropout=0.0,
            kdim=gene_emb_dim,
            vdim=gene_emb_dim,
            batch_first=True,
        )
        """
        self.encoder = XformersMultiheadAttention(
            gene_emb_dim,
            num_heads,
            dropout=0.0,
        )
        self.decoder = XformersMultiheadAttention(
            gene_emb_dim,
            num_heads,
            dropout=0.0,
        )

        self.mlp0 = MLP(query_dim, query_dim, dropout=dropout)
        self.mlp1 = MLP(gene_emb_dim, gene_emb_dim, dropout=dropout)

    def forward(
            self,
            latent_query: torch.Tensor,
            gene_query: torch.Tensor,
            key_val: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            residual: bool = True,
    ) -> torch.Tensor:
        norm_kv = self.layernorm_kv(key_val)
        norm_q = self.layernorm_q(latent_query)

        # Encoder: full input -> latent
        # latent_attn, _ = self.encoder(norm_q, norm_kv, norm_kv, key_padding_mask)
        latent_attn = self.encoder(norm_q, norm_kv, key_padding_mask)
        latent = latent_query + latent_attn
        latent = latent + self.mlp0(latent)

        # Decoder: latent -> full
        norm_g = self.layernorm_g(gene_query)
        norm_l = self.layernorm_l(latent)

        # decoder_out, _ = self.decoder(norm_g, norm_l, norm_l)
        decoder_out = self.decoder(norm_g, norm_l)
        gene_out = gene_query + decoder_out
        gene_out = key_val + self.mlp1(gene_out)

        return gene_out, latent


class DecoderNoCrossAttn(nn.Module):

    def __init__(
        self,
        seq_dim: int,
        dropout: float = 0.0,  # for the process attention module
        n_out: int = 1,
        layernorm: bool = True,
        hidden_expansion: float = 1,
        hidden_layers: int = 1,
    ):

        super().__init__()

        hidden_dim = int(hidden_expansion * seq_dim)

        seq = [
            nn.LayerNorm(seq_dim) if layernorm else nn.Identity(),
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        for n in range(1, hidden_layers):
            seq += [
                nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]

        output_layer = nn.Linear(hidden_dim, n_out)
        if n_out == 3:
            # Now manually initialize:
            # nn.init.xavier_uniform_(output_layer.weight, gain=0.1)  # or another sensible init
            nn.init.zeros_(output_layer.bias)

            # Then set pi bias to a negative value
            with torch.no_grad():
                output_layer.bias[2].fill_(-3.0)  # Or maybe -3.0 for even smaller dropout

        seq += [output_layer]

        self.gene_mlp = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor):

        gene_pred = self.gene_mlp(x)

        return gene_pred

class CellDecoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        cell_properties: Dict[str, Any],
        use_hidden_layer: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.cell_properties = cell_properties
        self.cell_mlp = nn.ModuleDict()
        self.batch_drop = nn.Dropout(p=float(dropout_rate))

        for k, cell_prop in cell_properties.items():
            # the output size of the cell property prediction MLP will be 1 if the property is continuous;
            # if it is discrete, then it will be the length of the possible values
            n_targets = 1 if not cell_prop["discrete"] else cell_prop["n_values"]

            if use_hidden_layer:
                print("Cell decoder hidden layer set to TRUE")
                self.cell_mlp[k] = nn.Sequential(
                    nn.Linear(latent_dim, 2 * self.latent_dim),
                    nn.ReLU(),
                    #nn.Dropout(p=float(dropout_rate)),
                    nn.Linear(2 * latent_dim, n_targets),
                )
            else:
                self.cell_mlp[k] = nn.Linear(latent_dim, n_targets, bias=True)


    def forward(self, latent: torch.Tensor):

        # Predict cell properties
        output = {}
        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
            x = latent.detach() if cell_prop["stop_grad"] else latent
            output[k] = self.cell_mlp[k](x)

        return output



class Decoder(nn.Module):

    def __init__(
            self,
            seq_dim: int,
            query_dim: int,
            cross_attn_dropout: float = 0.0,
            dropout: float = 0.0,  # for the process attention module
            n_out: int = 1,
            layernorm: bool = True,
            hidden_expansion: float = 1,
            hidden_layers: int = 1,
    ):

        super().__init__()

        self.decoder_cross_attn = CrossAttn(
            seq_dim,
            query_dim,
            dropout=cross_attn_dropout,
            residual=False,
        )  # query is now gene embedding

        hidden_dim = int(hidden_expansion * seq_dim)

        seq = [
            nn.LayerNorm(seq_dim) if layernorm else nn.Identity(),
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        for n in range(1, hidden_layers):
            seq += [
                nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]

        output_layer = nn.Linear(hidden_dim, n_out)
        if n_out == 3:
            # Now manually initialize:
            # nn.init.xavier_uniform_(output_layer.weight, gain=0.1)  # or another sensible init
            nn.init.zeros_(output_layer.bias)

            # Then set pi bias to a negative value
            with torch.no_grad():
                output_layer.bias[2].fill_(-3.0)  # Or maybe -3.0 for even smaller dropout

        seq += [output_layer]

        self.gene_mlp = nn.Sequential(*seq)

    def forward(self, latent: torch.Tensor, gene_query: torch.Tensor):

        # Query genes and cell properties
        # Decoder out will contain the latent for both genes and cell properties, concatenated together
        decoder_out = self.decoder_cross_attn(
            gene_query,
            latent,
            key_padding_mask=None,
        )

        # Predict genes
        n_genes = gene_query.size(1)
        gene_pred = self.gene_mlp(decoder_out[:, : n_genes, :])

        return gene_pred


class gMLP_stack(nn.Module):
    def __init__(self, seq_len: int, seq_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        self.gmlps = nn.Sequential(
            *[gMLP(seq_len, seq_dim, hidden_dim) for _ in range(n_layers)]
        )

        # self.gmlps = nn.Sequential(
        #    *[gMLP_fixedQ(seq_len, seq_dim, hidden_dim) for _ in range(n_layers)]
        # )

        # self.gmlps = nn.Sequential(
        #    *[GatedMLPWithMoE(seq_dim, hidden_dim, seq_len, num_experts=4) for _ in range(n_layers)]
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gmlps(x)


class hyena_stack(nn.Module):
    def __init__(self, seq_dim: int, n_layers: int, kernel_size: int = 64, hidden_dim: int = 256):
        super().__init__()

        self.hyena_seq = nn.Sequential(
            *[SimpleHyenaBlock(seq_dim, kernel_size=kernel_size, hidden_dim=hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hyena_seq(x)


class gMLP_fixedQ(nn.Module):
    def __init__(self, query_len: int, seq_dim: int, hidden_dim: int):
        super().__init__()

        # inputs is batch X seq_len X seq_dim
        self.ln0 = nn.LayerNorm(seq_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear0 = nn.Linear(seq_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, seq_dim)

        self.act = nn.GELU()
        gain = 0.01
        self.Q = nn.Parameter(torch.randn(query_len, query_len))
        torch.nn.init.xavier_normal_(self.Q, gain=gain)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.ln0(input)
        x = self.linear0(x)
        x = self.act(x)
        (u, v) = torch.chunk(x, 2, dim=-1)
        x = x + self.Q @ x
        x = self.linear2(x)

        return x + input


class gMLP(nn.Module):
    def __init__(self, query_len: int, seq_dim: int, hidden_dim: int):
        super().__init__()

        # inputs is batch X seq_len X seq_dim
        self.ln0 = nn.LayerNorm(seq_dim)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)
        self.linear0 = nn.Linear(seq_dim, hidden_dim)
        self.linear1 = nn.Linear(query_len, query_len)
        self.linear2 = nn.Linear(hidden_dim // 2, seq_dim)

        self.act = nn.GELU()

        torch.nn.init.ones_(self.linear1.bias)
        gain = 0.01
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        # self.pos_bias = nn.Parameter(gain * torch.randn(1, hidden_dim // 2, query_len))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.ln0(input)
        x = self.linear0(x)
        x = self.act(x)
        (u, v) = torch.chunk(x, 2, dim=-1)
        v = self.ln1(v)
        v = torch.transpose(v, 2, 1)
        v = self.linear1(v)  # + self.pos_bias
        v = torch.transpose(v, 2, 1)

        x = self.linear2(u * v)

        return x + input


class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_experts=4, k=1, dropout=0.0, noisy_gating=True):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.output_dim = output_dim or input_dim

        self.gate = nn.Linear(input_dim, num_experts)

        if hidden_dim > 0:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.output_dim)
                ) for _ in range(num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, self.output_dim),
                ) for _ in range(num_experts)
            ])

    def forward(self, x):
        """
        x: (B, N, D)
        returns: (B, N, output_dim)
        """
        B, N, D = x.shape
        # x_flat = x.view(B * N, D)  # (B*N, D)
        x_flat = x.reshape(B * N, D)  # (B*N, D)

        logits = self.gate(x_flat)  # (B*N, E)
        if self.training and self.noisy_gating:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise

        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        if self.k > 1:
            topk_weights = F.softmax(topk_vals, dim=-1)

        out_flat = torch.zeros(B * N, self.output_dim, device=x.device, dtype=x.dtype)

        if self.k == 1:
            for j, expert in enumerate(self.experts):
                mask = (topk_idx.view(-1) == j)
                if mask.any():
                    selected = x_flat[mask]
                    output = expert(selected)
                    out_flat[mask] = output.to(out_flat.dtype)  # no weight scaling
        else:
            for i in range(self.k):
                expert_id = topk_idx[:, i]
                weight = topk_weights[:, i].unsqueeze(1)

                for j, expert in enumerate(self.experts):
                    mask = (expert_id == j)
                    if mask.any():
                        selected = x_flat[mask]
                        output = expert(selected)
                        out_flat[mask] += output * weight[mask]

        return out_flat.view(B, N, self.output_dim)


class GatedMLPWithMoE(nn.Module):
    def __init__(self, dim, hidden_dim, seq_len, num_experts=4, top_k=1, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)

        # MoE returns a combined projection (U || V), so output_dim is 2×
        self.moe = MoELayer(
            input_dim=dim,
            hidden_dim=hidden_dim,
            output_dim=dim * 2,
            num_experts=num_experts,
            k=top_k,
            dropout=dropout,
            noisy_gating=True,
        )

        self.act = nn.GELU()
        gain = 0.01

        # Token mixing across sequence (N/latent length)
        self.pos_proj = nn.Linear(seq_len, seq_len)
        self.out_proj = nn.Linear(dim, dim)

        self.pos_bias = nn.Parameter(gain * torch.randn(1, dim, seq_len))

        torch.nn.init.ones_(self.pos_proj.bias)
        torch.nn.init.xavier_normal_(self.pos_proj.weight, gain=gain)

    def forward(self, x):
        """
        x: (B, N, D) — latent sequence
        """
        B, N, D = x.shape
        x_norm = self.norm(x)

        # Combined MoE output: (B, N, 2D)
        u_and_v = self.moe(x_norm)
        u_and_v = self.act(u_and_v)

        u, v = u_and_v.chunk(2, dim=-1)

        v = self.v_norm(v)

        # Positional mixing on v across tokens
        v = v.transpose(1, 2)  # (B, D, N)
        v = self.pos_proj(v) + self.pos_bias  # (B, D, N)
        v = v.transpose(1, 2)  # (B, N, D)

        return x + self.out_proj(u * v)  # Gated + residual


class ProcessSelfAttn(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            n_layers: int,
            n_heads: int,
            dim_feedforward: int,
            dropout: float = 0.0,
    ):
        super().__init__()
        print("ProcessSelfAttn drop", dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            n_heads,
            dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # NYM June 24
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.transformer(latent)

class FCLayers(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        layer_norm: bool = True,
    ):
        super().__init__()

        layers_dim = [n_in] + n_layers * [n_hidden]
        self.batch_drop = nn.Dropout(p=float(dropout_rate))

        layers = []
        for n in range(n_layers):
            layers.append(nn.Linear(layers_dim[n], layers_dim[n + 1]))
            layers.append(nn.GELU())
            if layer_norm:
                layers.append(nn.LayerNorm(layers_dim[n + 1], elementwise_affine=True))
            layers.append(nn.Dropout(p=float(dropout_rate)))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):

        return self.fc_layers(x)


class EncoderSCVI(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.0,
        input_dropout_rate: float = 0.0,
        shared_variance: bool = True,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, 1) if shared_variance else nn.Linear(n_hidden, n_output)
        self.softplus = nn.Softplus()

        self.drop = nn.Dropout(p=float(input_dropout_rate)) if input_dropout_rate > 0  else nn.Identity()

    def forward(self, x: torch.Tensor):

        x = self.drop(x)
        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = self.softplus(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


class DecoderSCVI(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        batch_properties: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.batch_properties = batch_properties

        n_additional_input = 0

        if batch_properties is not None:
            self.embedding = nn.ModuleDict([])
            for k in batch_properties.keys():
                n_vals = batch_properties[k]["n_values"]
                self.embedding[k] = self.gene_emb = nn.Embedding(n_vals + 1, n_input, padding_idx=None)
                n_additional_input += n_input
                print(k, "n_additional_input", n_additional_input)


        self.px_decoder = FCLayers(
            n_in=n_input + n_additional_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.theta = torch.nn.Parameter(torch.zeros(1, n_output))

        # mean
        self.px_mu = nn.Linear(n_hidden, n_output)

        # mean MSE
        self.px_mu_mse = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_drop = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_vals: Optional[torch.Tensor] = None,
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution

        if batch_vals is not None:
            for n, k in enumerate(self.batch_properties.keys()):
                v = self.embedding[k](batch_vals[:, n])
                z = torch.cat((z, v), dim=-1)

        z = self.px_decoder(z)
        px_mu = self.px_mu(z)
        px_mu_mse = self.px_mu_mse(z)
        px_drop = self.px_drop(z)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        #px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        #px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_mu, px_mu_mse, px_drop, self.theta

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()