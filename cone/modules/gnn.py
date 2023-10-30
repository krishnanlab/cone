from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn
from torch import Tensor
from torch_geometric.nn import GATv2Conv


class GNNLayer(nn.Module):

    def __init__(
        self,
        dim: int,
        *,
        dropout: float = 0.0,
        heads: int = 1,
    ):
        super().__init__()
        self.conv = GATv2Conv(dim, dim, heads=heads, concat=False)
        self.norm = pygnn.LayerNorm(dim, mode="node")
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.norm(x)
        x = self.conv(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GNN(nn.Module):

    def __init__(
        self,
        dim: int,
        num_layers: int,
        *,
        num_embs: Optional[int] = None,
        dropout: float = 0.0,
        heads: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Raw embeddings
        self.emb = nn.Embedding(num_embs, dim)

        # Message-passing layers
        self.mps = nn.ModuleList()
        for i in range(num_layers):
            self.mps.append(GNNLayer(dim, dropout=dropout, heads=heads))

        # Post-message-passing processor
        self.post_mp = torch.nn.Linear(dim, dim)

        self.reset_parameters()

    def reset_parameters(self):
        modules = self.modules()
        next(modules)  # skip self
        for m in modules:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        cond_emb: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.emb.weight
        if cond_emb is not None:
            x = x + cond_emb

        for i, mp in enumerate(self.mps):
            # Message passing with residual connection
            x = mp(x, edge_index, edge_weight=edge_weight) + x
        out = self.post_mp(x)

        return out
