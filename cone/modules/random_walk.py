"""
Modified from
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/node2vec.py
"""
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr

from cone.utils import get_num_workers


class RandomWalkLoader:
    def __init__(
        self,
        data: Data,
        *,
        walk_length: int = 80,
        context_size: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        scale_negative_loss: bool = True,
        num_workers: int = -1,
        rw_batch_size: int = 1024,
    ):
        # self.num_batches = rw_num_batches
        rw_loader_opts = dict(
            num_nodes=data.num_nodes,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
            num_negative_samples=num_negative_samples,
            scale_negative_loss=scale_negative_loss,
            batch_size=rw_batch_size,
        )

        self.rw_loader = RandomWalkSampler(
            data.edge_index,
            drop_last=True,
            num_workers=get_num_workers(num_workers),
            **rw_loader_opts,
        )

        self.ctxt_rw_loaders = {}
        for i, ctxt in enumerate(data.get("train_contexts", {})):
            ctxt_edge_index = data.edge_index[:, data.ctxt_edge_masks[i]]
            self.ctxt_rw_loaders[ctxt] = RandomWalkSampler(
                ctxt_edge_index,
                drop_last=False,
                # Use main process for loading rw from context graphs since
                # most of them only have a small number of nodes.
                num_workers=0,
                **rw_loader_opts,
            )

    def __len__(self):
        base_len = len(self.rw_loader.loader)
        return base_len if not self.ctxt_rw_loaders else base_len * 2

    def __getitem__(self, key):
        return self.rw_loader if key is None else self.ctxt_rw_loaders[key]

    def __iter__(self):
        for pos_rw, neg_rw in self.rw_loader:
            yield None, pos_rw, neg_rw

            if not self.ctxt_rw_loaders:  # no contexts
                continue

            ctxt = random.choice(list(self.ctxt_rw_loaders))
            ctxt_pos_rw, ctxt_neg_rw = next(iter(self.ctxt_rw_loaders[ctxt]))
            yield ctxt, ctxt_pos_rw, ctxt_neg_rw


class RandomWalkSampler(nn.Module):

    def __init__(
        self,
        edge_index: Tensor,
        walk_length: int,
        context_size: int,  # NOTE: half to the left and half to the right
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        scale_negative_loss: bool = True,
        num_nodes: Optional[int] = None,
        sparse: bool = False,
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__()

        if WITH_PYG_LIB and p == 1.0 and q == 1.0:
            self.random_walk_fn = torch.ops.pyg.random_walk
        elif WITH_TORCH_CLUSTER:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk
        else:
            if p == 1.0 and q == 1.0:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires either the 'pyg-lib' or "
                                  f"'torch-cluster' package")
            else:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires the 'torch-cluster' package")

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.neg_scale = num_negative_samples if scale_negative_loss else 1

        self.batch_size = batch_size
        self.valid_nodes = edge_index.unique()
        self.loader = DataLoader(
            self.valid_nodes,
            collate_fn=self.sample,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=num_workers > 1,
            pin_memory=True,
            **kwargs,
        )

    def __iter__(self):
        yield from self.loader

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(self.rowptr, self.col, batch,
                                 self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        # Draw negative samples by sampling random sequences of nodes as rw
        rand_idx = torch.randint(self.valid_nodes.numel(),
                                 (batch.size(0) * self.walk_length, ),
                                 dtype=batch.dtype, device=batch.device)
        rw = self.valid_nodes[rand_idx].view(-1, self.walk_length)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, z: Tensor, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""

        assert z.shape[0] == self.num_nodes
        if z.shape[0] != self.num_nodes:
            raise ValueError(
                f"Number of embedding vectors {z.shape[0]} is different from "
                f"the number of nodes in the graph {self.num_nodes}",
            )
        embedding_dim = z.shape[1]

        # Positive loss
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = z[start].view(pos_rw.size(0), 1, embedding_dim)
        h_rest = z[rest.view(-1)].view(pos_rw.size(0), -1, embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out).clamp(self.EPS)).mean()

        # Negative loss
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = z[start].view(neg_rw.size(0), 1, embedding_dim)
        h_rest = z[rest.view(-1)].view(neg_rw.size(0), -1, embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log((1 - torch.sigmoid(out)).clamp(self.EPS)).mean()

        return pos_loss + self.neg_scale * neg_loss

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}()")
