import itertools
from functools import reduce
from typing import Generator, Hashable

import numpy as np

from pyresolvemetrics._utils import _safe_division


def twi(ground_truth: frozenset[frozenset], result: frozenset[frozenset]) -> float:
    numerator = len(ground_truth) * len(result)
    overlap = reduce(
        lambda x, _: x + 1,
        filter(
            lambda intersection: len(intersection) > 0,
            itertools.starmap(
                lambda x, y: x & y, itertools.product(ground_truth, result)
            ),
        ),
        0,
    )
    denominator = overlap**2
    return numerator / denominator if denominator != 0 else 0


def _cluster_pairs(
    cluster: frozenset,
) -> Generator[tuple[Hashable, Hashable], None, None]:
    yield from itertools.combinations(cluster, 2)


def _comb_n_2(value: int) -> int:
    return (value * (value - 1)) // 2


def rand_index(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    contingency_table = np.array(
        [
            [len(gt_cluster & er_cluster) for er_cluster in result]
            for gt_cluster in ground_truth
        ],
        dtype=np.int32,
    )
    if len(np.shape(contingency_table)) == 1:
        return 0
    comb_2 = np.vectorize(_comb_n_2)
    tp_fp = np.sum(comb_2(np.sum(contingency_table, axis=0)))
    tp_fn = np.sum(comb_2(np.sum(contingency_table, axis=1)))
    tp = np.sum(comb_2(contingency_table))
    fp = tp_fp - tp
    fn = tp_fn - tp
    tn = _comb_n_2(np.sum(contingency_table)) - tp - fp - fn
    if (tp + tn + fp + fn) == 0:
        return 0

    return (tp + tn) / (tp + tn + fp + fn)


def adjusted_rand_index(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    initial_data_size = reduce(
        lambda count, cluster: count + len(cluster), ground_truth, 0
    )
    cn2 = _comb_n_2(initial_data_size)

    contingency_table = np.array(
        [
            [len(gt_cluster & er_cluster) for er_cluster in result]
            for gt_cluster in ground_truth
        ],
        dtype=np.int32,
    )
    a = np.sum(contingency_table, axis=1)
    b = np.sum(contingency_table, axis=0)
    comb_2 = np.vectorize(_comb_n_2)
    x = np.sum(comb_2(contingency_table))
    y = np.sum(comb_2(a))
    w = np.sum(comb_2(b))
    z = (y * w) / cn2
    if y + w == 2 * z:
        return 1.0
    ari = 2 * (x - z) / ((y + w) - 2 * z)

    return ari


def _partition_pairs(
    input_data: frozenset[frozenset],
) -> Generator[tuple[Hashable, Hashable], None, None]:
    yield from itertools.chain.from_iterable(map(_cluster_pairs, input_data))


def pair_precision(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    gt_pairs = set(_partition_pairs(ground_truth))
    res_pairs = set(_partition_pairs(result))
    return _safe_division(len(gt_pairs & res_pairs), len(res_pairs))


def pair_recall(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    gt_pairs = set(_partition_pairs(ground_truth))
    res_pairs = set(_partition_pairs(result))
    return _safe_division(len(gt_pairs & res_pairs), len(gt_pairs))


def pair_comparison_measure(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    pp = pair_precision(ground_truth, result)
    pr = pair_recall(ground_truth, result)
    return _safe_division(2 * pp * pr, pp + pr)


def cluster_precision(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    return _safe_division(len(ground_truth & result), len(result))


def cluster_recall(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    return _safe_division(len(ground_truth & result), len(ground_truth))


def cluster_comparison_measure(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    cp = cluster_precision(ground_truth, result)
    cr = cluster_recall(ground_truth, result)
    return _safe_division(2 * cp * cr, cp + cr)
