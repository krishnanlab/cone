from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.data import Data

from cone.modules import context_encoder
from cone.modules.gnn import GNN
from cone.modules.random_walk import RandomWalkLoader


class CONE(nn.Module):

    def __init__(self, model_cfg: DictConfig, data: Data):
        super().__init__()

        # Random walk loss
        rw_cfg = model_cfg.random_walk
        self.rw_loader = RandomWalkLoader(
            data,
            p=1,
            q=1,
            walk_length=rw_cfg.walk_length,
            context_size=rw_cfg.context_size,
            walks_per_node=rw_cfg.walks_per_node,
            num_negative_samples=rw_cfg.num_negative_samples,
            num_workers=rw_cfg.num_workers,
            rw_batch_size=rw_cfg.rw_batch_size,
        )

        # Context encoder
        ctxt_enc_cfg = model_cfg.context_encoder
        ctxt_enc_cls = getattr(context_encoder, ctxt_enc_cfg.type)
        self.cond_embs = ctxt_enc_cls(
            data,
            dim=model_cfg.dim,
            dim_hid=ctxt_enc_cfg.get("dim_hid"),
            num_layers=ctxt_enc_cfg.get("num_layers"),
            ctxt_sim=ctxt_enc_cfg.get("ctxt_sim"),
        )

        # Embedding decoder
        self.decoder = GNN(
            dim=model_cfg.dim,
            num_layers=model_cfg.num_layers,
            num_embs=data.num_nodes,
            dropout=model_cfg.dropout,
            heads=model_cfg.heads,
        )

    def forward(self, *args, ctxt: Optional[str] = None, **kwargs):
        cond_emb = self.cond_embs(ctxt)
        z = self.decoder(*args, **kwargs, cond_emb=cond_emb)
        return z

    def compute_loss(
        self,
        z: Tensor,
        pos_rw: Tensor,
        neg_rw: Tensor,
        *,
        ctxt: Optional[str] = None,
    ):
        return self.rw_loader[ctxt].loss(z, pos_rw, neg_rw)
