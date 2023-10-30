from itertools import combinations
from typing import Optional

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch_geometric.data import Data

from cone.modules.mlp import MLP


class BaseContextEncoder(nn.Module):
    def __init__(self, data: Data):
        super().__init__()
        self.contexts = data.get("contexts", [])
        self.ctxt_map = {j: i for i, j in enumerate(self.contexts)}

    def __iter__(self):
        yield from self.contexts

    def __getitem__(self, key: str) -> int:
        return self.ctxt_map[key]


class NullContextEncoder(BaseContextEncoder):
    def __init__(self, data: Data, **kwargs):
        super().__init__(data)

    def forward(self, ctxt: Optional[str] = None):
        assert ctxt is None, "NullContextEncoder only supports base context."
        return None


class EmbeddingContextEncoder(BaseContextEncoder):
    def __init__(self, data: Data, dim: int, **kwargs):
        super().__init__(data)
        self.embs = nn.Embedding(len(self.contexts), dim)

    def forward(self, ctxt: Optional[str] = None):
        if ctxt is not None:
            idx = self[ctxt]
            ctxt = self.embs.weight[idx]
        return ctxt


class MLPContextEncoder(BaseContextEncoder):
    def __init__(
        self,
        data: Data,
        dim: int,
        *,
        dim_hid: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        layernorm: bool = True,
        ctxt_sim: str = "jaccard",
        **kwargs,
    ):
        super().__init__(data)
        dim_hid = dim_hid or dim

        if not hasattr(data, "ctxt_spec_gene_masks"):
            return  # skip setting up since no context is available

        num_train_ctxt = len(data.train_contexts)

        # Calculate context similarity matrix
        x = data.ctxt_spec_gene_masks.float()
        if ctxt_sim == "jaccard":
            x_comm, x_diff = x @ x.T, x @ (1 - x.T)
            ctxt_sim_mat = x_comm / (x_comm + x_diff + x_diff.T)
        elif ctxt_sim == "cosine":
            x_normed = x / (x * x).sum(dim=1, keepdim=True).sqrt()
            ctxt_sim_mat = x_normed @ x_normed.T
        elif ctxt_sim == "rbf":  # NOTE: we use the median dist as 2 * sigmat^2
            x_norm = (x * x).sum(dim=1, keepdim=True)
            dist_mat = x_norm + x_norm.T - 2 * x @ x.T
            norm_factor = dist_mat[:num_train_ctxt, :num_train_ctxt].median()
            ctxt_sim_mat = torch.exp(- dist_mat / norm_factor)
        elif ctxt_sim == "spearman":
            ctxt_sim_mat = torch.eye(x.shape[0])
            for i, j in combinations(range(x.shape[0]), 2):
                ctxt_sim_mat[i, j] = ctxt_sim_mat[j, i] = spearmanr(x[i], x[j])[0]
        else:
            raise ValueError(f"Unknwon context similarity option {ctxt_sim!r}")
        ctxt_sim_mat = ctxt_sim_mat[:, :num_train_ctxt]
        self.register_buffer("ctxt_sim_mat", ctxt_sim_mat)

        # Set up processing MLP
        self.mlp = MLP(num_train_ctxt, dim, dim_hid, num_layers, dropout, layernorm)

    def forward(self, ctxt: Optional[str] = None):
        if ctxt is not None:
            assert hasattr(self, "mlp"), (
                "Context encoder not set up properly, which is likely due to "
                "missing 'ctxt_spec_gene_masks in the data.",
            )
            idx = self[ctxt]
            ctxt = self.mlp(self.ctxt_sim_mat)[idx]
        return ctxt
