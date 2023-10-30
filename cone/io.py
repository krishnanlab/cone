from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from torch import Tensor
from torch_geometric.data import Data as PyGData


def dump_ctxt_nets(
    network: str,
    context: str,
    data: PyGData,
    sep: str = " ",
    base_name: str = "base",
):
    from cone.config import PROCESSED_DATADIR, ensure_dir

    is_weighted = not (data.edge_weight == 1).all()

    def dump_net(path: Path, edge_mask: Optional[Tensor] = None):
        if edge_mask is None:
            edge_index, edge_weight = data.edge_index, data.edge_weight
        else:
            edge_index = data.edge_index[:, edge_mask]
            edge_weight = data.edge_weight[edge_mask]

        with open(path, "w") as f:
            for (i, j), k in zip(edge_index.T, edge_weight):
                line = [data.node_ids[i], data.node_ids[j]]
                if is_weighted:
                    line.append(str(k.item()))
                f.write(sep.join(line) + "\n")

        print(f"Network saved: {path}")

    dump_dir = ensure_dir(PROCESSED_DATADIR / "cone_ctxt_nets" / f"{network}-{context}")
    dump_net(dump_dir / f"{base_name}.txt")
    for ctxt, edge_mask in zip(data.contexts, data.ctxt_edge_masks):
        dump_net(dump_dir / f"{ctxt}.txt", edge_mask)


def read_column_from_csv(
    path: str,
    column: str,
    sep: str = ",",
    header: Optional[str] = "infer",
    unique: bool = True,
) -> List[str]:
    df = pd.read_csv(path, sep=sep, header=header)
    col = df[column]
    return col.unique().tolist() if unique else col.tolist()


def read_tissue_specific_genes_from_csv(path: str, key: str = "gene") -> List[str]:
    df = pd.read_csv(path)
    genes = df[key].unique().astype(str).tolist()
    return genes


def save_gmt_from_list(
    data: List[Dict[str, Union[str, List[str]]]],
    path: str,
    term_id_key: str = "term_id",
    term_name_key: str = "term_name",
    term_data_key: str = "term_data",
):
    with open(path, "w") as f:
        for term in data:
            term_id = term.get(term_id_key)
            term_name = term.get(term_name_key, term_id)
            term_data = term.get(term_data_key)
            line = "\t".join([term_id, term_name] + term_data)
            f.write(f"{line}\n")
