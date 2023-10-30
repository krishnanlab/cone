"""Evaluate embeddings on a particular gene set collection.

The embeddings are either precomputed (e.g., CONE dumps), or computed upon
calling this script, e.g., node2vec, or other precomputed baseline embeddings.

The results will be saved to the output results directory, and the specific
file name is constructed based on the input and output, as well as some other
specific notes or tags.

Example run script:

    .. code-block::

        $ python evaluate_embeddings.py --mode cone --num_workers 4
        $   --emb_dir outputs/cone-pinppi-tissue_gtex_expr-default/dump/ \

        $ python evaluate_embeddings.py --mode gemini --num_workers 4
        $   --emb_dir baselines/gemini/outputs/cone-pinppi-tissue_gtex_expr.npz \

        $ python evaluate_embeddings.py --mode bionic --num_workers 4
        $   --emb_dir baselines/bionic/outputs/cone-pinppi-tissue_gtex_expr_features.tsv \

        $ python evaluate_embeddings.py --mode gat --num_workers 4
        $   --emb_dir baselines/gat/outputs/pinppi.npz \

"""
import os
import os.path as osp
from itertools import combinations
from functools import partial
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import hypergeom
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm, trange

import cone.config
import cone.metrics
from cone.data import GMTData
from cone.utils import get_num_workers

pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 150

SEED = 42

LABEL_FILE_PATH = cone.config.GSC_DIR / "disgenet.gmt"
LABEL_NAME = "disgenet"

METRIC_DICT = {
    "AUROC": partial(cone.metrics.auroc, reduce="mean"),
    "APOP": partial(cone.metrics.apop, reduce="mean"),
    "APR@5": partial(cone.metrics.apr5, reduce="mean"),
}


def load_embeddings(
    network: str,
    emb_dir: Optional[str],
    mode: str,
    num_workers: int,
    tag: Optional[str],
    which: str = "latest",
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    if mode == "cone":
        run_name = emb_dir.split("/dump")[0].split("/")[-1]
        if emb_dir is None:
            raise ValueError("cone mode selected, which requires emb_dir to be specified")
        run_name = run_name if tag is None else f"{run_name}-{tag}"

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
            emb_dict[osp.splitext(fn)[0]] = np.load(path)
        assert emb_dict, f"No embeddings foud in {emb_dir} for {which}"

    elif mode == "node2vec":
        try:
            import obnb.data
            import obnb.ext.pecanpy
            import obnb.graph

        except ModuleNotFoundError as e:
            raise ImportError("obnb not installed:\n\t$ pip install obnb\n") from e

        run_name = f"n2v_{network.lower()}"
        if network != "PINPPI":
            g = getattr(obnb.data, network)(root="../data/obnb")
        else:
            path = cone.config.DATADIR / "networks" / "pinppi.npz"
            g = obnb.graph.SparseGraph.from_npz(path, weighted=False)
        emb = obnb.ext.pecanpy.pecanpy_embed(g, mode="PreCompFirstOrder",
                                             workers=num_workers, verbose=True)

        ids = list(emb.ids)
        emb_dict = {"base": emb.mat}

    elif mode in ("gemini", "gat"):
        run_name = f"{mode}_{network.lower()}"
        emb = np.load(emb_dir)
        ids = list(emb["ids"])
        emb_dict = {"base": emb["mat"]}

    elif mode == "bionic":
        run_name = f"bionic_{network.lower()}"
        emb_df = pd.read_csv(emb_dir, sep="\t", index_col=0)
        ids = emb_df.index.astype(str).tolist()
        emb_dict = {"base": emb_df.values}

    else:
        raise NotImplementedError(f"Unknown mode {mode!r}")

    save_filename = f"{run_name}_{LABEL_NAME}.csv"

    return ids, emb_dict, save_filename


def load_labels(
    gene_ids: List[str],
    min_num_pos: int = 5,
    num_workers: int = 1,
) -> pd.DataFrame:
    print(f"Loading label from {LABEL_FILE_PATH}")
    gmt = GMTData.from_gmt(LABEL_FILE_PATH)
    df = gmt.to_df(gene_ids)
    # label_dict = {}
    # with open(LABEL_FILE_PATH) as f:  # FIX: use GMTData?
    #     for line in f:
    #         terms = line.rstrip().split("\t")
    #         label_dict[terms[0]] = sorted_intersect(terms[2:], gene_ids)
    # label_names = sorted(label_dict)

    # # Set up label matrix with postive examples
    # df = pd.DataFrame(index=gene_ids, columns=label_names).fillna(0)
    # for label_name, genes in label_dict.items():
    #     df.loc[genes, label_name] = 1

    # Remove tasks with insufficient positives
    orig_num = df.shape[1]
    df = df.loc[:, (df == 1).sum(0) >= min_num_pos]
    if orig_num != (filtered_num := df.shape[1]):
        print(
            f"Filtered terms based on positves ({min_num_pos=}): "
            f"{orig_num} -> {filtered_num}",
        )

    # Set up negatives
    neg_dict = setup_negative_hypergeom(df, num_workers=num_workers)
    for label_name, genes in neg_dict.items():
        df.loc[genes, label_name] = -1

    # Display task summary
    summary_df = pd.DataFrame(
        {
            "task": df.columns,
            "name": [gmt.get(i)[1] for i in df.columns],
            "num_positives": (df.values == 1).sum(0),
            "num_negatives": (df.values == -1).sum(0),
        }
    ).sort_values("num_positives", ascending=False).reset_index(drop=True)
    print(summary_df)

    return df


def setup_negative_hypergeom(
    df: pd.DataFrame,
    p_thresh: float = 0.05,
    num_workers: int = 1,
):
    num_tasks = df.shape[1]
    print(f"Setting up negatives ({num_tasks=})")

    calc_hyeprgeom_worker = partial(
        calc_hyeprgeom,
        label_mat=df.values,
        total=(df.values == 1).any(1).sum(),
    )

    jobs = tqdm(list(combinations(range(num_tasks), 2)))
    p = Parallel(n_jobs=num_workers)
    res = p(delayed(calc_hyeprgeom_worker)(i) for i in jobs)

    mat_pval = np.eye(num_tasks)
    for i, j, pval in res:
        mat_pval[i, j] = mat_pval[j, i] = pval

    neg_label = {}
    mat_ind = mat_pval < p_thresh
    use_idx = np.where((df.values == 1).any(1))[0]
    for i, (name, arr) in enumerate(zip(df.columns, df.values.T)):
        neg_idx = set(use_idx) - set(np.where(arr == 1)[0])

        if mat_ind[i].any():
            to_exclude = np.where(mat_ind[i])[0]
            neg_idx -= set(np.where(df.values[:, to_exclude].any(1))[0])

        neg_label[name] = sorted(df.iloc[list(neg_idx)].index)

    return neg_label


def calc_hyeprgeom(
    idx_tuple: Tuple[int, int],
    label_mat: np.ndarray,
    total: Optional[int] = None,
):
    arr1, arr2 = map(lambda x: label_mat[:, x], idx_tuple)
    num_common = np.logical_and(arr1 == 1, arr2 == 1).sum()

    if num_common > 0:
        total = (label_mat == 1).any(1).sum() if total is None else total
        pval = hypergeom.sf(
            num_common - 1,
            (arr1 == 1).sum(),
            (arr2 == 1).sum(),
            total,
        )
    else:
        pval = 1

    return *idx_tuple, pval


def evaluate_emb(
    mode: str,
    emb_dict: Dict[str, np.ndarray],
    use_ind: np.ndarray,
    y: np.ndarray,
    task_ids: List[str],
    n_trials: int = 10,
    num_workers: int = 1,
) -> pd.DataFrame:

    def update_results(x: np.ndarray, mode: str):
        # Run evaluations (tasks x runs)
        run_eval_worker = partial(eval_worker, x=x, y=y, seed=SEED, num_runs=n_trials)
        p = Parallel(n_jobs=num_workers)
        jobs = trange(y.shape[1], leave=False)
        pred_res = p(delayed(run_eval_worker)(i) for i in jobs)

        # Register results
        for i, task_id in enumerate(task_ids):
            for run_id, run_res in enumerate(pred_res[i]):
                for metric_name, metric_func in METRIC_DICT.items():
                    score_dict = {
                        split: metric_func(*run_res[split])
                        for split in ("train", "val", "test")
                    }
                    num_pos_genes = sum(i[0].sum() for i in run_res.values())
                    num_neg_genes = sum((i[0] == 0).sum() for i in run_res.values())

                    for split, score in score_dict.items():
                        results.append({
                            "task_id": task_id,
                            "context": context,
                            "mode": mode,
                            "num_pos_genes": num_pos_genes,
                            "num_neg_genes": num_neg_genes,
                            "split": split,
                            "score": score,
                            "score_type": metric_name,
                            "run_id": run_id,
                        })

    # Obtain base (context-naive) embeddings
    x_base = [j for i, j in emb_dict.items() if i.startswith("base.")]
    if x_base:
        assert len(x_base) == 1
        x_base = x_base[0][use_ind]
    else:
        x_base = None

    results = []
    pbar = tqdm(emb_dict.items(), total=len(emb_dict), desc="Evaluating embeddings")
    for context, emb in pbar:
        # Subset to labeled entries
        x = emb[use_ind]

        # Directly use embedding
        update_results(x, mode="single")

        if x_base is None:  # context-naive embeddings only
            continue

        # # Concatenate context-specific with context-naive
        # x_ = np.hstack((x, x_base))
        # update_results(x_, mode="concat")

        # Project concatenated with PCA to match dimension
        x_ = PCA(n_components=x.shape[1]).fit_transform(np.hstack((x, x_base)))
        update_results(x_, mode="concat-pca")

    return pd.DataFrame(results)


def eval_worker(i: int, x: np.ndarray, y: np.ndarray, seed: int, num_runs: int = 10):
    res = []
    for run_id in range(num_runs):
        if (y[:, i] == -1).any():  # prepare using specificed negatives
            use_ind = y[:, i] != 0
            x_use, y_use = x[use_ind], y[use_ind, i].clip(0)
        else:
            x_use, y_use = x, y[:, i]

        xs, ys = stratify_split(x_use, y_use, seed=seed + run_id)
        clf = LogisticRegression(penalty="l2", C=0.2, max_iter=2000, n_jobs=1)
        clf.fit(xs[0], ys[0])
        res.append({
            "train": (ys[0], clf.decision_function(xs[0])),
            "val": (ys[1], clf.decision_function(xs[1])),
            "test": (ys[2], clf.decision_function(xs[2])),
        })

    return res


def stratify_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    pos_idx, neg_idx = map(lambda i: np.where(y == i)[0], (1, 0))
    n1, n2 = map(lambda i: int(pos_idx.size * i), (train_ratio, train_ratio + val_ratio))
    n3, n4 = map(lambda i: int(neg_idx.size * i), (train_ratio, train_ratio + val_ratio))
    pos_split_idxs = np.split(rng.permutation(pos_idx.size), (n1, n2))
    neg_split_idxs = np.split(rng.permutation(neg_idx.size), (n3, n4))
    assert all(i.size for i in pos_split_idxs), "Missing positives in a split"

    split_x, split_y = [], []
    for i, j in zip(pos_split_idxs, neg_split_idxs):
        split_x.append(np.concatenate((x[pos_idx[i]], x[neg_idx[j]])))
        split_y.append(np.concatenate((y[pos_idx[i]], y[neg_idx[j]])))

    return split_x, split_y


def display_summary(
    df: pd.DataFrame,
    target_metric="APOP",
    target_split: str = "test",
):
    task_summary_group = ["task_id", "context"]
    run_summary_group = ["run_id", "context"]
    if "mode" in df.columns:
        task_summary_group.append("mode")
        run_summary_group.append("mode")

    # Summarize across tasks
    task_summary_df = (
        df.query("score_type == @target_metric & split == @target_split")
        .groupby(task_summary_group, as_index=False)
        .mean(numeric_only=True)
        .drop(columns="run_id")
        .pivot_table("score", "task_id", "context")
    )
    print(task_summary_df.describe())

    # Summarize across runs
    run_summary_df = (
        df.query("score_type == @target_metric & split == @target_split")
        .groupby(run_summary_group, as_index=False)
        .mean(numeric_only=True)
        .pivot_table("score", "run_id", "context")
    )
    for ctxt, scores in run_summary_df.T.iterrows():
        avg, std = scores.mean(), scores.std()
        print(f"{ctxt} ({target_split} {target_metric}): {avg:.3f} +/- {std:.3f}")


@click.command()
@click.option("--network", default="PINPPI", type=click.Choice(["PINPPI"]))
@click.option("--mode", default="cone",
              type=click.Choice(["cone", "node2vec", "bionic", "gemini", "gat"]))
@click.option("--emb_dir", type=click.Path(), default=None)
@click.option("--which", type=str, default="latest")
@click.option("--n_trials", type=int, default=5)
@click.option("--tag", type=str, default=None)
@click.option("--num_workers", type=int, default=-1)
@click.option("--save_dir", type=click.Path(exists=True), default="results/")
def main(
    network: str,
    emb_dir: Optional[str],
    mode: str,
    n_trials: int,
    num_workers: int,
    tag: Optional[str],
    which: str,
    save_dir: str,
):
    num_workers = get_num_workers(num_workers)

    # Prepare data for evaluation
    ids, emb_dict, save_filename = load_embeddings(
        network, emb_dir, mode, num_workers, tag, which)
    label_df = load_labels(ids, num_workers=num_workers)

    out_path = osp.join(save_dir, save_filename)
    print(f"Results will be saved to {out_path}")

    task_ids = label_df.columns.tolist()
    use_ind = label_df.values.any(1) | (label_df.values == -1).any(1)
    y = label_df.values[use_ind].astype(int)

    # Main evaluation loop
    results_df = evaluate_emb(
        mode, emb_dict, use_ind, y, task_ids, n_trials, num_workers)

    # Summarize and dump results
    display_summary(results_df)
    results_df.to_csv(out_path, index=False)
    print(f"Reulst saved to {out_path}")


if __name__ == "__main__":
    main()
