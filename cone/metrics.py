from functools import wraps, partial
from typing import Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def ensure_numpy_2d_array(obj: Union[np.ndarray, torch.Tensor]):
    # Ensure type
    if isinstance(obj, np.ndarray):
        new_obj = obj
    elif isinstance(obj, torch.Tensor):
        new_obj = obj.detach().cpu().numpy()
    else:
        raise ValueError(
            "Unkown type encountered during type casting to numpy "
            f"arrays for metrics clculation: {type(obj)!r}",
        )

    # Ensure shape
    if len(new_obj.shape) == 1:
        new_obj = new_obj[:, None]
    elif len(new_obj.shape) != 2:
        raise ValueError(f"Invalid shape encountered {new_obj.shape}")

    return new_obj


def metrics_decorator(metric_func):
    """Decorator to perform common operations for metrics.

    1. Cast input types to 2-d numpy arry.
    2. Add reduce option.

    """

    @wraps(metric_func)
    def bounded_metric_func(
        true: Union[np.ndarray, torch.Tensor],
        pred: Union[np.ndarray, torch.Tensor],
        reduce: str = "none",
    ) -> Union[np.ndarray, float]:
        res = metric_func(
            ensure_numpy_2d_array(true),
            ensure_numpy_2d_array(pred),
        )

        if reduce == "mean":
            res = res.mean()
        elif reduce != "none":
            raise ValueError(
                f"Unknown reduction method {reduce!r}. Available options "
                "are: ['mean', 'none']"
            )

        return res

    return bounded_metric_func


@metrics_decorator
def apop(true: np.ndarray, pred: np.ndarray):
    """Log2 average precision over prior."""
    prior = true.mean(0)
    ap = average_precision_score(true, pred, average=None)
    return np.log2(ap / prior)


@metrics_decorator
def ap(true: np.ndarray, pred: np.ndarray):
    """Average precision score."""
    return average_precision_score(true, pred, average=None)


@metrics_decorator
def acc(true: np.ndarray, pred: np.ndarray):
    """Accuracy score (assume predictions are probabilities)."""
    assert true.shape[1] == pred.shape[1] == 1, "Only support single class now"
    res = ((true > 0) == (pred > 0.5)).mean()
    return np.array([res])


@metrics_decorator
def bacc(true: np.ndarray, pred: np.ndarray):
    """Balanced accuracy score (assumes predictions are probabilities)."""
    assert true.shape[1] == pred.shape[1] == 1, "Only support single class now"
    pos_acc = (pred[true > 0] > 0.5).mean()
    neg_acc = (pred[true <= 0] <= 0.5).mean()
    res = (pos_acc + neg_acc) / 2
    return np.array([res])


@metrics_decorator
def auroc(true: np.ndarray, pred: np.ndarray):
    """AUROC score."""
    return roc_auc_score(true, pred, average=None)


@metrics_decorator
def aprk(true: np.ndarray, pred: np.ndarray, k: int):
    """Average Precision and Recall @ K.

    See section 6.5 https://www.biorxiv.org/content/10.1101/2023.07.18.549602v1

    Note:
        Our implementation clips k to be the total number of positives.

    """
    assert true.shape[1] == pred.shape[1] == 1, "Only support single class now"
    true, pred = true.ravel(), pred.ravel()
    sorted_idx = (-pred).argsort()  # sort ascendingly

    k = min(k, true.sum())  # clip by number of positives
    sorted_true_k_bool = true[sorted_idx[:k]] == 1

    precision_at_k = sorted_true_k_bool.cumsum() / (np.arange(k) + 1)
    res = (precision_at_k * sorted_true_k_bool).mean()

    return np.array([res])


apr5 = metrics_decorator(partial(aprk.__wrapped__, k=5))
apr10 = metrics_decorator(partial(aprk.__wrapped__, k=10))
apr20 = metrics_decorator(partial(aprk.__wrapped__, k=20))


if __name__ == "__main__":
    a = np.array([0, 1, 2])
    target = [[0], [1], [2]]

    assert ensure_numpy_2d_array(a).tolist() == target
    assert ensure_numpy_2d_array(a[:, None]).tolist() == target
    assert isinstance(ensure_numpy_2d_array(torch.Tensor(a)), np.ndarray)
    assert ensure_numpy_2d_array(torch.Tensor(a)).tolist() == target
