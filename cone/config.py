__all__ = [
    "EPS",
    "HOMEDIR",
    "LOSS_TYPE",
]

from pathlib import Path
from typing import Literal

from cone.utils import ensure_dir

HOMEDIR = Path(__file__).resolve().parents[2]
RESULTDIR = ensure_dir(HOMEDIR / "results")
DATADIR = ensure_dir(HOMEDIR / "data")
RAW_DATADIR = ensure_dir(DATADIR / "raw")
PROCESSED_DATADIR = ensure_dir(DATADIR / "processed")
EXTRAS_DATADIR = ensure_dir(DATADIR / "extras")

# Processed data paths
GSC_DIR = ensure_dir(PROCESSED_DATADIR / "gene_set_collections")
CTXT_DIR = ensure_dir(GSC_DIR / "contexts")  # context gene lists

EPS = 1e-12
LOSS_TYPE = Literal["binary", "weighted_l1", "weighted_l2"]
BASE_CONDITION_NAME = "base"
