# from typing import Dict, List, Tuple
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from cone.data import ContextData, NetworkData
from cone.utils import ids_to_map

pd.options.display.max_rows = 1000


def load_data(cfg: DictConfig) -> Data:
    network = NetworkData(cfg.network)
    context = ContextData(cfg.context)
    data = contextualize_network(network, context)
    print(f"[*] Data constructed:\n{data}")
    return data


def contextualize_network(network, context):
    data = network.to_pyg_data()

    if context is None or context.ctxt_mode == "none":
        return data

    mode = context.ctxt_mode  # spec, none
    network_node_idmap = ids_to_map(data.node_ids)
    contexts = context.contexts
    train_contexts = context.train_contexts

    ctxt_spec_gene_masks = torch.zeros(len(contexts), len(network_node_idmap), dtype=bool)
    for i, ctxt in enumerate(contexts):
        ids = context.ctxt_spec_genes_dict[ctxt]
        idx = list(filter(None, map(network_node_idmap.get, ids)))
        ctxt_spec_gene_masks[i, idx] = True

    ctxt_edge_masks = get_ctxt_edge_masks(
        data.edge_index,
        ctxt_spec_gene_masks,
        mode,
    )

    def format(x, num=True):
        return [f"{i:,}" if num else f"{i:.2%}" for i in x.tolist()]

    print(f"Contextualization mode: {mode!r}")
    stats = pd.DataFrame(
        {
            "Context": contexts,
            "# genes": format(ctxt_spec_gene_masks.sum(1)),
            "# edges": format(ctxt_edge_masks.sum(1)),
            "% edges": format(
                ctxt_edge_masks.sum(1) / ctxt_edge_masks.shape[1],
                num=False,
            ),
        }
    )
    print(stats)

    data.ctxt_spec_gene_masks = ctxt_spec_gene_masks
    data.ctxt_edge_masks = ctxt_edge_masks
    data.contexts = contexts
    data.train_contexts = train_contexts  # subset (or all) of contexts used for training

    return data


def get_ctxt_edge_masks(
    edge_index: torch.Tensor,
    ctxt_spec_gene_masks: torch.Tensor,
    mode: str = "spec",
) -> torch.Tensor:
    assert mode == "spec"
    assert len(ctxt_spec_gene_masks.shape) == 2  # context x genes

    def edge_mask_getter(
        edge_index: torch.Tensor,
        ctxt_spec_gene_mask: torch.Tensor,
    ) -> torch.Tensor:
        ctxt_spec_gene_idx = torch.where(ctxt_spec_gene_mask)[0]
        return subgraph(ctxt_spec_gene_idx, edge_index, return_edge_mask=True)[-1]

    ctxt_edge_masks = torch.zeros(
        ctxt_spec_gene_masks.shape[0],
        edge_index.shape[1],
        dtype=bool,
    )
    for ctxt_idx in range(ctxt_spec_gene_masks.shape[0]):
        ctxt_edge_masks[ctxt_idx] = edge_mask_getter(
            edge_index,
            ctxt_spec_gene_masks[ctxt_idx],
        )

    return ctxt_edge_masks


def load_gmt(path: str) -> pd.DataFrame:
    ids, genelists = [], []
    with open(path) as f:
        for line in f:
            # [term_id, term_info, term_gene_1, term_gene_2, ..., term_gene_n]
            terms = line.rstrip().split("\t")
            ids.append(terms[0])
            genelists.append(terms[2:])

    mlb = MultiLabelBinarizer()
    encodings = mlb.fit_transform(genelists)

    # gene x term
    label_df = pd.DataFrame(encodings, index=ids, columns=mlb.classes_).T

    return label_df


def load_edgelist_txt(path: str) -> Tuple[sp.csr_matrix, List[str]]:
    g = nx.read_weighted_edgelist(path, delimiter=" ")
    data = nx.to_scipy_sparse_array(g)
    node_ids = list(g.nodes)
    return data, node_ids


def load_edgelist_npz(path: str) -> Tuple[sp.csr_matrix, List[str]]:
    npz = np.load(path)

    node_ids = npz["node_ids"].tolist()
    edge_index = npz["edge_index"]

    if "edge_weight" in npz.files:
        print(f"Loading weighted graph from {path}")
        edge_weight = npz["edge_weight"]
    else:
        print(f"Loading unweighted graph from {path}")
        edge_weight = np.ones_like(edge_index[0], dtype=float)

    data = sp.coo_matrix((edge_weight, (edge_index[0], edge_index[1])))

    return data, node_ids
