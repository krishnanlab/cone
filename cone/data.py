from itertools import chain
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data


class GMTData:
    TERM_ID_KEY = "term_id"
    TERM_NAME_KEY = "term_name"
    TERM_DATA_KEY = "term_data"

    def __init__(self):
        # E.g., [{"term_id": "MONDO:12345", "term_name": "disease name",
        #         "term_data": ["123", "2344", "3456"]}, ...]
        self._data = []
        self._id_to_idx = {}

    def __len__(self) -> int:
        return len(self._data)

    @property
    def num_terms(self) -> int:
        return len(self)

    @property
    def terms(self) -> List[str]:
        return list(self._id_to_idx)

    @property
    def entities(self) -> List[str]:
        return sorted(set(chain.from_iterable(i[self.TERM_DATA_KEY] for i in self._data)))

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    def __repr__(self) -> str:
        return f"GMTData(num_terms: {len(self):,}, num_entities: {self.num_entities:,})"

    def __iter__(self):
        for term_id in self._id_to_idx:
            yield self.get(term_id)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, term_id) -> Union[str, str, List[str]]:
        idx = self._id_to_idx[term_id]
        data = self._data[idx]
        return data[self.TERM_ID_KEY], data[self.TERM_NAME_KEY], data[self.TERM_DATA_KEY]

    def add(self, term_id: str, term_name: str, term_data: List[str]):
        if term_id in self._id_to_idx:
            raise KeyError(f"{term_id=} already exists")
        self._id_to_idx[term_id] = len(self._id_to_idx)
        self._data.append(
            {
                self.TERM_ID_KEY: term_id,
                self.TERM_NAME_KEY: term_name,
                self.TERM_DATA_KEY: term_data,
            }
        )

    def update(
        self,
        term_id: str,
        *,
        term_name: Optional[str] = None,
        term_data: Optional[str] = None,
        create_on_miss: bool = False,
    ):
        """Update term.

        Args:
            term_id: ID for the term to update.
            term_name: If supplied, will overwrite the previous setting of
                term name associated with this ID.
            term_data: If supplied, will update the term data by combing the
                previous list with the supplied list by taking the set of
                items and then sort into a list.
            create_on_miss: If set to True, then create a new entry (by calling
                :meth:`add` when the specified term_id does not exist yet),
                which requires both term_name and term_data to be set.
                Otherwise, raise KeyError if term_id does not exists yet.

        """
        if (idx := self._id_to_idx.get(term_id)) is None:
            if create_on_miss:
                assert (term_name is not None) and (term_data is not None)
                self.add(term_id, term_name, term_data)
                return
            else:
                raise KeyError(f"{term_id=} does not exist")

        if term_name is not None:
            self._data[idx][self.TERM_NAME_KEY] = term_name

        if term_data is not None:
            old_term_data = self._data[idx][self.TERM_DATA_KEY]
            merged_term_data = sorted(set(term_data + old_term_data))
            self._data[idx][self.TERM_DATA_KEY] = merged_term_data

    def read_gmt(self, path: str, exists_ok: bool = False):
        with open(path) as f:
            for line in f:
                data = line.rstrip().split("\t")
                if exists_ok:
                    self.update(data[0], term_name="N/A", term_data=data[2:],
                                create_on_miss=True)
                else:
                    self.add(data[0], data[1], data[2:])

    def save_gmt(self, path: str):
        with open(path, "w") as f:
            for term_id in sorted(self._id_to_idx):
                term_id, term_name, term_data = self.get(term_id)
                line = "\t".join([term_id, term_name] + term_data)
                f.write(f"{line}\n")

    def to_df(self, selected_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Set up label matrix with postive examples marked as 1s.

        Each column is a term (gene set) and each row is a gene.

        Args:
            selected_ids: Specified gene ordering. If not specified, then use
                all genes present in the gene set collection ordered
                alphabetically as the gene ordering. Note that this also act
                as filtering (only use genes that are present in the selected
                ids list).

        """
        from cone.utils import sorted_intersect

        selected_ids = selected_ids or self.entities
        df = pd.DataFrame(index=selected_ids, columns=self.terms).fillna(0)
        for term_id, _, term_data in self:
            filtered_term_data = sorted_intersect(term_data, selected_ids)
            df.loc[filtered_term_data, term_id] = 1

        return df

    @classmethod
    def from_gmt(cls, path: str, exists_ok: bool = False):
        gmt = cls()
        gmt.read_gmt(path, exists_ok=exists_ok)
        return gmt


class NetworkData:

    def __init__(self, network_cfg: DictConfig):
        from cone import loaders

        if (data_loader := getattr(loaders, network_cfg.loader)) is None:
            raise ValueError(f"Unknown loader {network_cfg.loader!r}")
        data, self.node_ids = data_loader(network_cfg.path)

        if isinstance(data, np.ndarray):
            assert data.shape[0] == data.shape[1]
            self._adj = data
        elif isinstance(data, sp.spmatrix):
            assert data.shape[0] == data.shape[1]
            self._edge_index = data.tocoo()
        else:
            raise TypeError(f"Unsupported data type {type(data)}")

    @property
    def adj(self) -> np.ndarray:
        return self._adj if self.is_dense else self._edge_index.toarray()

    @property
    def edge_index(self) -> sp.coo_matrix:
        return self._edge_index if self.is_sparse else sp.coo_matrix(self._adj)

    @property
    def is_dense(self) -> bool:
        return hasattr(self, "_adj")

    @property
    def is_sparse(self) -> bool:
        return hasattr(self, "_edge_index")

    @property
    def node_ids(self) -> List[str]:
        return self._node_ids

    @node_ids.setter
    def node_ids(self, val: List[str]):
        if not isinstance(val, list) or any(not isinstance(i, str) for i in val):
            raise TypeError("node_ids must be a list of string")
        self._node_ids = val

    @property
    def size(self) -> int:
        len(self.node_ids)

    def to_dense(self):
        if not self.is_dense:
            self._adj = self.adj

    def to_sparse(self):
        if not self.is_sparse:
            self._edge_index = self._edge_index

    def to_pyg_data(self):
        edge_index = torch.LongTensor(np.vstack(np.nonzero(self.edge_index)))
        edge_weight = torch.FloatTensor(self.edge_index.data)
        return Data(
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_ids=self.node_ids,
            num_nodes=self.size,
        )

    @classmethod
    def from_nx(cls, g: nx.Graph):
        edge_index = nx.to_scipy_sparse_array(g).tocoo()
        node_ids = list(g.nodes)
        return cls(edge_index, node_ids)


class ContextData:

    def __init__(self, context_cfg: DictConfig):
        self.ctxt_mode = context_cfg.mode
        if self.ctxt_mode != "none":
            self.load_ctxt_spec_genes(context_cfg.ctxt_spec_genes)

    @property
    def ctxt_mode(self) -> str:
        return self._ctxt_mode

    @ctxt_mode.setter
    def ctxt_mode(self, mode: str):
        avail_opts = ["spec", "none"]
        if mode not in avail_opts:
            raise ValueError(
                f"Unknown mode {mode!r}, available options are {avail_opts}",
            )
        self._ctxt_mode = mode

    @property
    def ctxt_spec_genes_dict(self) -> Dict[str, List[str]]:
        return self._ctxt_spec_genes_dict

    @property
    def contexts(self) -> List[str]:
        return self._contexts

    @property
    def train_contexts(self) -> List[str]:
        return self._train_contexts

    def load_ctxt_spec_genes(self, ctxt_spec_genes_cfg: DictConfig):
        path = ctxt_spec_genes_cfg.path
        loader = ctxt_spec_genes_cfg.loader
        ctxt_key = ctxt_spec_genes_cfg.get("ctxt_key")
        node_key = ctxt_spec_genes_cfg.get("node_key")

        # Specify contexts of interest
        selected_ctxts = set(ctxt_spec_genes_cfg.get("selected_ctxts") or {})
        holdout_ctxts = set(ctxt_spec_genes_cfg.get("holdout_ctxts") or {})
        if holdout_ctxts and not selected_ctxts:
            raise ValueError(
                "Must specify training contexts when holdout context is specified.",
            )
        if (common := holdout_ctxts & selected_ctxts):
            raise ValueError(
                f"Holdout context must not appear in training contexts: {common}",
            )
        all_ctxts = selected_ctxts | holdout_ctxts

        # Load context gene sets
        if loader == "csv":
            df = pd.read_csv(path)
            df[node_key] = df[node_key].astype(str)  # cast str type to Entrez

            if all_ctxts:
                df = df[df[ctxt_key].isin(all_ctxts)].reset_index(drop=True)
                print(f"Using contexts: {sorted(all_ctxts)}")
            else:
                print(f"No contexts specified, using all: {sorted(df[ctxt_key].unique())}")

            self._ctxt_spec_genes_dict = {}
            for ctxt, group in df.groupby(ctxt_key):
                self._ctxt_spec_genes_dict[ctxt] = sorted(set(group[node_key]))

        elif loader == "gmt":
            gmt = GMTData.from_gmt(path)

            if all_ctxts:
                if (diff := all_ctxts.difference(set(gmt.terms))):
                    raise ValueError(f"Unknown specified contexts {sorted(diff)}")
                print(f"Using contexts: {sorted(all_ctxts)}")
            else:
                all_ctxts = gmt.terms
                print(f"No contexts specified, using all: {sorted(all_ctxts)}")

            self._ctxt_spec_genes_dict = {i: gmt.get(i)[2] for i in all_ctxts}

        else:
            raise ValueError(f"Unknown context gene loader: {loader!r}")

        # Set context loaded
        loaded_ctxts = set(self.ctxt_spec_genes_dict)
        loaded_holdout_ctxts = sorted(loaded_ctxts.intersection(holdout_ctxts))
        self._train_contexts = sorted(loaded_ctxts.difference(holdout_ctxts))
        self._contexts = self._train_contexts + sorted(loaded_holdout_ctxts)
        print(f"Total number of contexts: {len(self._contexts)}")

        if loaded_holdout_ctxts:
            print(f"Number of training contexts: {len(self._train_contexts)}")
            print(f"Number of held-out contexts: {len(loaded_holdout_ctxts)}")
            print(f"Held-out contexts: {loaded_holdout_ctxts}")
