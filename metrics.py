import numpy as np
import torch
import sklearn.metrics
import scipy


def single_label_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).sum().item() / preds.size(0)


def multi_label_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).all(-1).sum().item() / preds.size(0)


def tokens_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, ignore_id: int = -100
) -> float:
    return (
        (preds[labels != ignore_id] == labels[labels != ignore_id]).sum()
        / (labels != ignore_id).sum()
    ).item()


def f1_score(preds: torch.Tensor, labels: torch.Tensor, average: str) -> float:
    return sklearn.metrics.f1_score(
        labels.view(-1).numpy(), preds.view(-1).numpy(), average=average
    )


def matthews_correlation_coefficient(
    preds: torch.Tensor, labels: torch.Tensor
) -> float:
    return sklearn.metrics.matthews_corrcoef(
        labels.view(-1).numpy(), preds.view(-1).numpy()
    )


def pearson_correlation(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return scipy.stats.pearsonr(preds.view(-1).numpy(), labels.view(-1).numpy())[0]


def spearman_correlation(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return scipy.stats.spearmanr(preds.view(-1).numpy(), labels.view(-1).numpy())[0]


def ppl(loss: float) -> float:
    return np.exp(loss)
