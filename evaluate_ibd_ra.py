"""Evaluate embedding performance on the two PINNACLE drug tasks.

The results will be saved to the specified output results directory, with the
following name format: ${run_name}-pinnacle_drug_targets.csv

run_name is interpolated from the emb_dir argument by stripping the trailing
``/dump/`` and the anything before the last ``/``, e.g., the run_name for
``outputs/cone-pinppi-celltype_pinnacle-default/dump/`` would be
``cone-pinppi-celltype_pinnacle-default``.

Example run script:

    .. code-block::

        $ python evaluate_ibd_ra.py  --mode cone \
        $   --emb_dir outputs/cone-pinppi-celltype_pinnacle-default/dump/ \
        $   --num_workers 4 --subset_pinnacle_genes

"""
import json
import os
import os.path as osp
from functools import partial
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import cone.config
import cone.metrics
from cone.utils import ensure_dir, get_num_workers, sanitize_emb_name, sorted_intersect

pd.options.display.max_rows = 1000

LABEL_NAME = "pinnacle_drug_targets"
LABEL_FILE_PATH = cone.config.GSC_DIR / f"{LABEL_NAME}.json"
PINNACLE_EMB_DIR = cone.config.HOMEDIR / "baselines" / "pinnacle" / "outputs" / "pinnacle"

METRIC_DICT = {
    "APOP": partial(cone.metrics.apop, reduce="mean"),
    "APR@5": partial(cone.metrics.apr5, reduce="mean"),
}

POS_KEY = "positive"
NEG_KEY = "negative"
TRAIN_KEY = "train"
TEST_KEY = "test"

TRAIN_IND_VAL = 1
TEST_IND_VAL = 2


def load_pinnacle_embeddings() -> Dict[str, Dict[str, np.ndarray]]:
    pinnacle_emb_dict = {}
    for path in glob(osp.join(str(PINNACLE_EMB_DIR), "*")):
        print(f"Loading embedding from {path}")
        name = path.split("/")[-1].replace(".npz", "")
        file = np.load(path)
        pinnacle_emb_dict[name] = {
            "node_ids": file["node_ids"].tolist(),
            "emb": file["emb"],
        }
    return pinnacle_emb_dict


def load_embeddings(
    network: str,
    emb_dir: Optional[str],
    mode: str,
    num_workers: int,
    pinnacle_emb_dict: Optional[Dict[str, Dict[str, Union[List[str], np.ndarray]]]],
    which: str = "latest",
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    if mode == "cone":
        run_name = emb_dir.split("/dump")[0].split("/")[-1]
        if emb_dir is None:
            raise ValueError("cone mode selected, which requires emb_dir to be specified")

        try:
            ids = np.load(osp.join(emb_dir, "ids.npy")).tolist()
        except FileNotFoundError:
            ids = np.load(osp.join(emb_dir, "node_ids.npy")).tolist()

        emb_dict = {}
        emb_dir = osp.join(emb_dir, "embeddings")
        emb_dir = osp.join(emb_dir, "latest") if which == "latest" else emb_dir
        for fn in os.listdir(emb_dir):
            if not fn.endswith(f".{which}.npy"):
                continue
            path = osp.join(emb_dir, fn)
            print(f"Loading {path}")
            name = sanitize_emb_name(osp.splitext(fn)[0])
            emb_dict[name] = np.load(path)
        assert emb_dict, f"No embeddings foud in {emb_dir} for {which}"

    elif mode == "pinnacle":
        run_name = "pinnacle"
        emb_dict = {}
        ids = set()
        for i in pinnacle_emb_dict.values():
            ids.update(i["node_ids"])
        ids = list(ids)

    else:
        raise NotImplementedError(f"Unknown mode {mode!r}")

    save_filename = f"{run_name}-{LABEL_NAME}.csv"

    return ids, emb_dict, save_filename


def load_labels(
    gene_ids: List[str],
    min_num_pos: int = 5,
    num_workers: int = 1,
) -> pd.DataFrame:
    print(f"Loading label from {LABEL_FILE_PATH}")

    with open(LABEL_FILE_PATH) as f:
        label_data = json.load(f)
    label_names = sorted(label_data)

    label_df = pd.DataFrame(index=gene_ids, columns=label_names).fillna(0)
    split_df = pd.DataFrame(index=gene_ids, columns=label_names).fillna(0)
    for name in label_names:
        data = label_data[name]
        pos, neg, tr, ts = map(data.get, (POS_KEY, NEG_KEY, TRAIN_KEY, TEST_KEY))

        label_df.loc[sorted_intersect(pos, gene_ids), name] = 1
        label_df.loc[sorted_intersect(neg, gene_ids), name] = -1

        split_df.loc[sorted_intersect(tr, gene_ids), name] = TRAIN_IND_VAL
        split_df.loc[sorted_intersect(ts, gene_ids), name] = TEST_IND_VAL

    # Display task summary
    summary_df = pd.DataFrame(
        {
            "task": label_df.columns,
            "num_positives": (label_df.values == 1).sum(0),
            "num_negatives": (label_df.values == -1).sum(0),
        }
    ).sort_values("num_positives", ascending=False).reset_index(drop=True)
    print(summary_df)

    return label_df, split_df


def evaluate_emb(
    ids: List[str],
    mode: str,
    emb_dict: Dict[str, np.ndarray],
    label_df: pd.DataFrame,
    train_mask_df: pd.DataFrame,
    test_mask_df: pd.DataFrame,
    task_ids: List[str],
    pinnacle_emb_dict: Optional[Dict[str, Dict[str, Union[List[str], np.ndarray]]]],
    subset_pinnacle_genes: bool,
    num_workers: int = 1,
    run_id: int = 0,
) -> pd.DataFrame:

    def update_results(cone_mode: str):
        # Run evaluations (tasks x runs)
        run_eval_worker = partial(
            eval_worker,
            ids=ids,
            label_df=label_df,
            train_mask_df=train_mask_df,
            test_mask_df=test_mask_df,
            emb_dict=emb_dict,
            emb_base=emb_base,
            pinnacle_emb_dict=pinnacle_emb_dict,
            subset_pinnacle_genes=subset_pinnacle_genes,
            cone_mode=cone_mode,
            run_mode=mode,
            seed=42,
        )

        # Prepare context list for evluation (pinnacle or cone)
        contexts = sorted(emb_dict) or sorted(pinnacle_emb_dict)

        # Evaluate different contexts in parallel
        jobs = tqdm(contexts, desc=f"Evaluating {cone_mode} mode")
        p = Parallel(n_jobs=num_workers)
        pred_res = p(delayed(run_eval_worker)(i) for i in jobs)

        # Register results
        for context, run_res in zip(contexts, pred_res):
            for task_id, task_res in zip(task_ids, run_res):
                for metric_name, metric_func in METRIC_DICT.items():
                    score_dict = {
                        split: metric_func(*task_res[split])
                        for split in ("train", "test")
                    }
                    num_pos_genes = sum(i[0].sum() for i in task_res.values())
                    num_neg_genes = sum((i[0] == 0).sum() for i in task_res.values())

                    for split, score in score_dict.items():
                        results.append({
                            "task_id": task_id,
                            "context": context,
                            "mode": cone_mode,
                            "num_pos_genes": num_pos_genes,
                            "num_neg_genes": num_neg_genes,
                            "split": split,
                            "score": score,
                            "score_type": metric_name,
                            "run_id": run_id,
                        })

    # Obtain base (context-naive) embeddings
    emb_base = emb_dict.get("base")

    results = []
    update_results(cone_mode="single")
    if emb_base is not None:
        update_results(cone_mode="concat")
        update_results(cone_mode="concat-pca")

    return pd.DataFrame(results)


def eval_worker(
    ctxt_name: str,
    ids: List[str],
    label_df: pd.DataFrame,
    train_mask_df: pd.DataFrame,
    test_mask_df: pd.DataFrame,
    # use_ind_series: pd.Series,
    emb_dict: Dict[str, np.ndarray],
    emb_base: Optional[np.ndarray],
    pinnacle_emb_dict: Optional[Dict[str, Dict[str, Union[List[str], np.ndarray]]]],
    subset_pinnacle_genes: bool,
    cone_mode: str,
    run_mode: str,
    seed: int,
):
    if run_mode == "pinnacle":
        assert cone_mode == "single"
        x = pinnacle_emb_dict[ctxt_name]["emb"]
    else:
        x = emb_dict[ctxt_name]

    if cone_mode.startswith("concat"):
        x = np.hstack((x, emb_base))
        if cone_mode.endswith("pca"):
            x = PCA(n_components=x.shape[1]).fit_transform(x)
    elif cone_mode != "single":
        raise ValueError(f"Unknwon cone_mode {cone_mode!r}")

    if subset_pinnacle_genes and ctxt_name != "base":
        pinnacle_ids = pinnacle_emb_dict[ctxt_name]["node_ids"]
        label_df = label_df.reindex(pinnacle_ids)
        train_mask_df = train_mask_df.reindex(pinnacle_ids)
        test_mask_df = test_mask_df.reindex(pinnacle_ids)
        if run_mode != "pinnacle":
            x = pd.DataFrame(x, index=ids).reindex(pinnacle_ids).values
    y = label_df.values
    train_mask = train_mask_df.values
    test_mask = test_mask_df.values

    res = []
    for i in range(label_df.shape[1]):
        use_ind = y[:, i] != 0
        x_use, y_use = x[use_ind], y[use_ind, i].clip(0)
        train_mask_use, test_mask_use = train_mask[use_ind, i], test_mask[use_ind, i]

        x_train, y_train = x_use[train_mask_use], y_use[train_mask_use]
        x_test, y_test = x_use[test_mask_use], y_use[test_mask_use]

        clf = LogisticRegression(penalty="l2", C=0.2, max_iter=2000, n_jobs=1)
        clf.fit(x_train, y_train)
        res.append({
            "train": (y_train, clf.decision_function(x_train)),
            "test": (y_test, clf.decision_function(x_test)),
        })

    return res


@click.command()
@click.option("--network", default="PINPPI", type=click.Choice(["PINPPI"]))
# Whether subset pinnacle celltype specific genes for evaluation (must be True for pinnacle)
@click.option("--subset_pinnacle_genes", is_flag=True)
@click.option("--mode", default="cone",
              type=click.Choice(["cone", "pinnacle"]))
@click.option("--emb_dir", type=click.Path(), default=None)
@click.option("--which", type=str, default="latest")
@click.option("--num_workers", type=int, default=-1)
@click.option("--run_id", type=int, default=0)
@click.option("--save_dir", type=click.Path(), default="results/")
def main(
    network: str,
    subset_pinnacle_genes: bool,
    emb_dir: Optional[str],
    mode: str,
    num_workers: int,
    which: str,
    run_id: int,
    save_dir: str,
):
    num_workers = get_num_workers(num_workers)
    print(f"{subset_pinnacle_genes=}")
    if subset_pinnacle_genes or mode == "pinnacle":
        if mode == "pinnacle":
            assert subset_pinnacle_genes
        pinnacle_emb_dict = load_pinnacle_embeddings()
    else:
        pinnacle_emb_dict = None

    ids, emb_dict, save_filename = load_embeddings(network, emb_dir, mode, num_workers,
                                                   pinnacle_emb_dict, which)
    label_df, split_df = load_labels(ids, num_workers=num_workers)

    ensure_dir(save_dir)
    out_path = osp.join(save_dir, save_filename)
    print(f"Results will be saved to {out_path}")

    task_ids = label_df.columns.tolist()
    train_mask_df = split_df == TRAIN_IND_VAL
    test_mask_df = split_df == TEST_IND_VAL

    results_df = evaluate_emb(ids, mode, emb_dict, label_df, train_mask_df,
                              test_mask_df, task_ids, num_workers=num_workers,
                              run_id=run_id, pinnacle_emb_dict=pinnacle_emb_dict,
                              subset_pinnacle_genes=subset_pinnacle_genes)

    for score_type in METRIC_DICT:
        print(f"\n{score_type} summary:")
        summry_results_df = (
            results_df
            .query("score_type == @score_type")
            .pivot_table("score", ["task_id", "context", "mode"], "split")
            .reset_index()
            .sort_values("test", ascending=False)
            .groupby("task_id")
            .head(5)
        )
        print(summry_results_df)

    results_df.to_csv(out_path, index=False)
    print(f"Reulst saved to {out_path}")


if __name__ == "__main__":
    main()
