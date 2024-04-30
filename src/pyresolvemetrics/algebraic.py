import itertools
from functools import reduce
from typing import Generator

import numpy as np


def twi(ground_truth: list[set], result: list[set]) -> float:
    numerator = len(ground_truth) * len(result)
    overlap = 0
    for gt_set in ground_truth:
        for res_set in result:
            if len(res_set & gt_set) > 0:
                overlap += 1
    denominator = overlap**2
    return numerator / denominator if denominator != 0 else 0


def _get_unique_identifiers(partition: list[set[tuple]]) -> list[int]:
    result = {}
    for cluster in partition:
        key = frozenset(cluster)
        result[key] = hash(key)
    return list(result.values())


def _cluster_pairs(cluster: set[tuple]) -> Generator[tuple, None, None]:
    yield from itertools.combinations(cluster, 2)


def _comb_n_2(value: int) -> int:
    return (value * (value - 1)) // 2


def rand_index(ground_truth: list[set], result: list[set]) -> float:
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


def adjusted_rand_index(ground_truth: list[set], result: list[set]) -> float:
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
    input_data: list[set[tuple]],
) -> Generator[tuple[tuple], None, None]:
    for cluster in input_data:
        yield from _cluster_pairs(cluster)


def pair_precision(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    gt_pairs = set(_partition_pairs(ground_truth))
    res_pairs = set(_partition_pairs(result))
    if len(res_pairs) == 0:
        # no pairs were retrieved -> precision = 0
        return 0.0
    return len(gt_pairs & res_pairs) / (len(res_pairs))


def pair_recall(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    gt_pairs = set(_partition_pairs(ground_truth))
    res_pairs = set(_partition_pairs(result))
    return len(gt_pairs & res_pairs) / (len(gt_pairs))


def pair_comparison_measure(
    ground_truth: list[set[tuple]], result: list[set[tuple]]
) -> float:
    pp = pair_precision(ground_truth, result)
    pr = pair_recall(ground_truth, result)
    if pp + pr == 0:
        return 0.0

    return (2 * pp * pr) / (pp + pr)


def _cluster(input_data: list[set[tuple]]) -> Generator[tuple[tuple], None, None]:
    return (tuple(v for v in partition_class) for partition_class in input_data)


def cluster_precision(
    ground_truth: list[set[tuple]], result: list[set[tuple]]
) -> float:
    gt_cluster = set(_cluster(ground_truth))
    res_cluster = set(_cluster(result))
    if len(res_cluster) == 0:
        # if no clusters were retrieved, the precision is zero
        return 0

    return len(gt_cluster & res_cluster) / len(res_cluster)


def cluster_recall(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    gt_cluster = set(_cluster(ground_truth))
    res_cluster = set(_cluster(result))

    return len(gt_cluster & res_cluster) / len(gt_cluster)


def cluster_comparison_measure(
    ground_truth: list[set[tuple]], result: list[set[tuple]]
) -> float:
    cp = cluster_precision(ground_truth, result)
    cr = cluster_recall(ground_truth, result)
    if cp + cr == 0:
        return 0.0

    return (2 * cp * cr) / (cp + cr)
