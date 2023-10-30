import importlib
import itertools
import os
import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Sequence, Set

import mygene
import networkx as nx
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from torch_geometric.data import Data

from cone.data import GMTData


def sorted_intersect(seq1, seq2):
    return sorted(set(seq1) & set(seq2))


def sanitize_emb_name(name: str) -> str:
    return ".".join(name.split(".")[:-1])


def sanitize_name(name: str) -> str:
    return name.replace("/", "|")


def convert_gmt_genes_to_entrez(gmt: GMTData, source_type: str) -> GMTData:
    if source_type not in ("symbol", "ensembl.gene"):
        warnings.warn(
            f"{source_type} not tested yet, there might be unexpected behaviors",
            UserWarning,
            stacklevel=2,
        )

    mapping_dict = get_to_entrez_mapping(gmt.entities, source_type)
    mapped_genes = set(mapping_dict)

    gmt_entrez = GMTData()
    for term_id, term_name, term_genes in gmt:
        if (common_genes := set(term_genes) & mapped_genes):
            converted_term_genes = sorted(
                set(
                    itertools.chain.from_iterable(
                        mapping_dict[i]
                        for i in common_genes
                    )
                )
            )
            gmt_entrez.add(term_id, term_name, converted_term_genes)

    return gmt_entrez


def get_to_entrez_mapping(
    gene_ids,
    source_type: str = "symbol",
    bijective_only: bool = False,
) -> Dict[str, Set[str]]:
    print(f"Total number of genes ({source_type}) to query: {len(gene_ids):,}")

    mg = mygene.MyGeneInfo()
    mapping_df = mg.querymany(gene_ids, species="human", scopes=source_type,
                              fields="entrezgene", as_dataframe=True)
    mapping_df = mapping_df[~pd.isna(mapping_df["entrezgene"])].reset_index()

    mapping_dict, inv_mapping_dict = defaultdict(set), defaultdict(set)
    for i, j in mapping_df[["query", "entrezgene"]].values:
        mapping_dict[i].add(j)
        inv_mapping_dict[j].add(i)

    if bijective_only:
        source_to_remove = {i for i, j in mapping_dict.items() if len(j) > 1}
        target_to_remove = {i for i, j in inv_mapping_dict.items() if len(j) > 1}

        for source in source_to_remove:
            mapping_dict.pop(source)

        mapping_dict = {
            i: j for i, j in mapping_dict.items()
            if (i not in source_to_remove) and (j not in target_to_remove)
        }

    return dict(mapping_dict)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def count_params(m: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the total numnber of (trainable) parameters in model."""
    params = filter(lambda p: p.requires_grad, m.parameters())
    return sum([np.prod(p.size()) for p in params])


def get_func_from_config(
    config: DictConfig,
    *,
    key_prefix: str = "",
    func_name_key: str = "_func_",
    func_kwargs_key: str = "_func_kwargs_",
    force_check_kwargs: bool = False,
) -> Callable:
    """Load function with kwargs set."""
    # Example: read -> _read
    key_prefix = f"_{key_prefix.lstrip('_')}"
    # Example: _func_ -> _read_func_
    func_name_key = f"{key_prefix}_{func_name_key.lstrip('_')}"
    # Example: _func_kwargs_ -> _read_func_kwargs_
    func_kwargs_key = f"{key_prefix}_{func_kwargs_key.lstrip('_')}"

    if (func_str := config.get(func_name_key, None)) is None:
        raise ValueError(
            f"{func_name_key!r} is unavailable in the input config {config}.\n"
            f"Please make sure you specified the correct func_name_key.",
        )

    func = get_func_from_str(func_str)
    if (func_kwargs := config.get(func_kwargs_key, None)) is None:
        if force_check_kwargs:
            raise ValueError(
                f"{func_kwargs_key!r} is unavailable in the input config "
                f"{config}.\nPlease make sure you specified the correct "
                "func_kwargs_key, or disable force_check_kwargs if you do not "
                "need kwargs for the function.",
            )
        func_kwargs = {}

    return partial(func, **func_kwargs)


def get_func_from_str(func_str: str) -> Callable:
    """Load function from a string.

    First load the module (inferred as everything up to the last '.'). Then
    return the function inside that module.

    This is effectively the same as the followig:

        .. codeblock::

            >>> func_str = "a.b.c"
            >>> import a.b
            >>> return a.b.c

    """
    parts = func_str.split(".")
    module_name = ".".join(parts[:-1])
    func_name = parts[-1]

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Attempt to load {func_str!r} failed due to unknown module {module_name!r}",
        ) from e

    try:
        func = getattr(module, func_name)
    except AttributeError as e:
        raise AttributeError(
            f"Attempt to load {func_str!r} failed due to unknown function "
            f"{func_name} in module {module_name}"
        ) from e

    return func


def maybe_log_wandb(data: Dict[str, Any]):
    """Log data to wandb if wandb is set up."""
    if wandb.run is not None:
        wandb.run.log(data)


def idx_to_mask(idx: torch.Tensor, size: int):
    mask = idx.new_zeros(size, dtype=bool)
    mask[idx] = True
    return mask


def nx_to_pyg_data(g: nx.Graph) -> Data:
    node_ids = list(g.nodes)
    adj_coo = nx.to_scipy_sparse_array(g, format="coo")

    edge_index = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
    edge_weight = torch.FloatTensor(adj_coo.data)

    data = Data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_ids=node_ids,
        num_nodes=g.number_of_nodes(),
    )

    return data


def get_device(device: str) -> torch:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto selected device: {device}")
    return device


def get_num_workers(num_workers: int, check_int_type: bool = True) -> int:
    """Infer the maximum number of workers using if -1."""
    if check_int_type and not isinstance(num_workers, int):
        raise TypeError(f"num_workers must be int type, got {type(num_workers)}")

    if num_workers == -1:
        try:
            import numba

            num_workers = numba.config.NUMBA_DEFAULT_NUM_THREADS
        except ModuleNotFoundError:
            import multiprocessing

            num_workers = multiprocessing.cpu_count()

    return num_workers


def ids_to_map(ids: Sequence[str]) -> Dict[str, int]:
    """Return id to index mapping from a sequence of ids."""
    return {j: i for i, j in enumerate(ids)}
