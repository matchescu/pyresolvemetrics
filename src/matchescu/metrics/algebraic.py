import itertools
from math import sqrt
from typing import Iterable, Generator

from numpy import ndarray, zeros, sum, vectorize


def extract_algebraic_result_model(input_data: Iterable[Iterable[Iterable]]) -> list[set[tuple]]:
    return [
        set(
            tuple(v for v in partition_item)
            for partition_item in partition_class
        )
        for partition_class in input_data
    ]


def _compute_partition_overlap(
        ground_truth: list[set[tuple]], result: list[set[tuple]]
) -> list[set[tuple]]:
    retval = []
    for i in ground_truth:
        for j in result:
            overlap = i & j
            if len(overlap) < 1:
                continue
            retval.append(overlap)
    return retval


def _compute_intersection_matrix(
        ground_truth: list[set[tuple]], result: list[set[tuple]]
) -> ndarray:
    retval = zeros((len(ground_truth), len(result)), dtype=int)
    for i, class_i in enumerate(ground_truth):
        for j, class_j in enumerate(result):
            retval[i, j] = len(class_i & class_j)
    return retval


def _compute_cn2(n: int) -> int:
    return (n * (n - 1)) // 2


def _compute_total_number_of_elements(ground_truth: list[set[tuple]]) -> int:
    reconstructed = set()
    for partition_class in ground_truth:
        reconstructed |= partition_class
    return len(reconstructed)


def twi(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    partition_overlap = _compute_partition_overlap(ground_truth, result)
    return sqrt(len(ground_truth) * len(result)) / len(partition_overlap)


def _compute_rand_index_components(
        ground_truth: list[set[tuple]], result: list[set[tuple]]
) -> tuple[int, int, int, int]:
    intersection_matrix = _compute_intersection_matrix(ground_truth, result)
    si = list(map(len, ground_truth))
    sj = list(map(len, result))
    smn = _compute_total_number_of_elements(ground_truth)

    cn2 = vectorize(_compute_cn2)

    x = sum(cn2(intersection_matrix))
    y = sum(cn2(si))
    z = sum(cn2(sj))

    return x, y, z, cn2(smn) - x - y - z


def rand_index(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    x, y, z, w = _compute_rand_index_components(ground_truth, result)
    return (x + w) / (x + y + z + w)


def adjusted_rand_index(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    x, y, z, w = _compute_rand_index_components(ground_truth, result)
    qty = ((y + x) * (z + x)) / (x + y + z + w)

    return (x - qty) / (((y + z + 2 * x) / 2) - qty)


def _pairs(input_data: list[set[tuple]]) -> Generator[tuple[tuple], None, None]:
    for partition_class in input_data:
        for pair in itertools.combinations(partition_class, 2):
            yield pair


def pair_precision(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    gt_pairs = set(_pairs(ground_truth))
    res_pairs = set(_pairs(result))
    return len(gt_pairs & res_pairs) / (len(gt_pairs))


def pair_recall(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    gt_pairs = set(_pairs(ground_truth))
    res_pairs = set(_pairs(result))
    if len(res_pairs) == 0:
        return 0.0
    return len(gt_pairs & res_pairs) / (len(res_pairs))


def pair_comparison_measure(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    pp = pair_precision(ground_truth, result)
    pr = pair_recall(ground_truth, result)
    if pp + pr == 0:
        return 0.0

    return (2 * pp * pr) / (pp + pr)


def _cluster(input_data: list[set[tuple]]) -> Generator[tuple[tuple], None, None]:
    return (
        tuple(v for v in partition_class) for partition_class in input_data
    )


def cluster_precision(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    gt_cluster = set(_cluster(ground_truth))
    res_cluster = set(_cluster(result))

    return len(gt_cluster & res_cluster) / len(gt_cluster)


def cluster_recall(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    gt_cluster = set(_cluster(ground_truth))
    res_cluster = set(_cluster(result))

    return len(gt_cluster & res_cluster) / len(res_cluster)


def cluster_comparison_measure(ground_truth: list[set[tuple]], result: list[set[tuple]]) -> float:
    cp = cluster_precision(ground_truth, result)
    cr = cluster_recall(ground_truth, result)

    return (2 * cp * cr) / (cp + cr)
