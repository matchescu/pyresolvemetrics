import itertools
from functools import reduce
from typing import Generator, Hashable

import numpy as np

from pyresolvemetrics._utils import _safe_division


def twi(ground_truth: frozenset[frozenset], result: frozenset[frozenset]) -> float:
    """Compute the Talburt-Wang index.

    The Talburt-Wang Index (TWI) evaluates the similarity between two partitions
    by considering the number of overlapping subsets. It is calculated as the
    ratio of the number of overlaps to the total number of subsets in both
    partitions.

    :param ground_truth: a set of sets. Represents the ideal algebraic partition
        induced by the entity resolution relation over an algebraic set.
    :param result: a set of sets. Represents the partition produced by the
        entity resolution task over the same algebraic set as the ground truth.

    :returns: a floating point value between 0.0 and 1.0. A TWI of 1 indicates
        identical partitions, while lower values suggest lesser degrees of
        similarity.
    """
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
    r"""Compute the Rand index.

    The Rand Index is a measure of similarity between two data clusterings,
    assessing the agreement of element pairs. It calculates the proportion of
    pairs that are either clustered together or separately in both partitions
    relative to all possible pairs.

    :param ground_truth: a set of sets. Represents the ideal algebraic partition
        induced by the entity resolution relation over an algebraic set.
    :param result: a set of sets. Represents the partition produced by the
        entity resolution task over the same algebraic set as the ground truth.
    :returns: a floating point value between 0.0 and 1.0. A Rand index of 1
        indicates perfect pairwise agreement.
    """

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
    """Compute the adjusted Rand index.

    The Adjusted Rand Index (ARI) modifies the Rand Index to account for chance
    grouping. It adjusts the baseline of the Rand Index, ensuring that random
    label assignments yield an expected ARI of 0. This adjustment provides a
    more accurate assessment of clustering similarity, especially when dealing
    with random or arbitrary partitions.

    :param ground_truth: a set of sets. Represents the ideal algebraic partition
        induced by the entity resolution relation over an algebraic set.
    :param result: a set of sets. Represents the partition produced by the
        entity resolution task over the same algebraic set as the ground truth.

    :returns: a floating point value between -1.0 and 1.0. An ARI of 1
        indicates perfect pairwise agreement. Negative values indicate that the
        agreement between clusters is less than the one expected to occur
        through random chance (i.e. there is a significant disagreement).
    """
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
    r"""Compute the pair precision over entity resolution clusters.

    Pair precision is defined as the number of pairwise combinations of elements
    that are common between the ``ground_truth`` (G) and the ``result`` (R),
    divided by the number of pairwise combinations of elements in `R`.

    .. math::

        |Pairs(G) \cap Pairs(R)| \over |Pairs(R)|

    :param ground_truth: a set of sets. Represents the ideal algebraic partition
        induced by the entity resolution relation over an algebraic set.
    :param result: a set of sets. Represents the partition produced by the
        entity resolution task over the same algebraic set as the ground truth.
    """
    gt_pairs = set(_partition_pairs(ground_truth))
    res_pairs = set(_partition_pairs(result))
    return _safe_division(len(gt_pairs & res_pairs), len(res_pairs))


def pair_recall(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    r"""Compute the pair recall over entity resolution clusters.

    Pair recall is defined as the number of pairwise combinations of elements
    that are common between the ``ground_truth`` (G) and the ``result`` (R),
    divided by the number of pairwise combinations of elements in `G`.

    .. math::

        |Pairs(G) \cap Pairs(R)| \over |Pairs(G)|

    :param ground_truth: a set of sets. Represents the ideal algebraic partition
        induced by the entity resolution relation over an algebraic set.
    :param result: a set of sets. Represents the partition produced by the
        entity resolution task over the same algebraic set as the ground truth.
    """
    gt_pairs = set(_partition_pairs(ground_truth))
    res_pairs = set(_partition_pairs(result))
    return _safe_division(len(gt_pairs & res_pairs), len(gt_pairs))


def pair_comparison_measure(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    r"""Compute the pair comparison measure.

    The pair comparison measure is defined as the harmonic mean between pair
    precision (pp) and pair recall (pr).

    .. math::

        2 \cdot pp \cdot pr \over pp + pr

    :param ground_truth: a set of sets. Represents the ideal algebraic partition
        induced by the entity resolution relation over an algebraic set.
    :param result: a set of sets. Represents the partition produced by the
        entity resolution task over the same algebraic set as the ground truth.
    """
    pp = pair_precision(ground_truth, result)
    pr = pair_recall(ground_truth, result)
    return _safe_division(2 * pp * pr, pp + pr)


def cluster_precision(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    r"""Compute the cluster precision over entity resolution clusters.

    Cluster precision is defined as the number of common clusters between
    the ``ground_truth`` (G) and the ``result`` (R), divided by the number of
    clusters in `R`.

    .. math::

        |Clusters(G) \cap Clusters(R)| \over |Clusters(R)|

    :param ground_truth: a set of sets. Represents the ideal algebraic partition
        induced by the entity resolution relation over an algebraic set.
    :param result: a set of sets. Represents the partition produced by the
        entity resolution task over the same algebraic set as the ground truth.
    """
    return _safe_division(len(ground_truth & result), len(result))


def cluster_recall(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    r"""Compute the cluster recall over entity resolution clusters.

    Cluster recall is defined as the number of common clusters between
    the ``ground_truth`` (G) and the ``result`` (R), divided by the number of
    clusters in `G`.

    .. math::

        |Clusters(G) \cap Clusters(R)| \over |Clusters(G)|

    :param ground_truth: a set of sets. Represents the ideal algebraic partition
        induced by the entity resolution relation over an algebraic set.
    :param result: a set of sets. Represents the partition produced by the
        entity resolution task over the same algebraic set as the ground truth.
    """
    return _safe_division(len(ground_truth & result), len(ground_truth))


def cluster_comparison_measure(
    ground_truth: frozenset[frozenset], result: frozenset[frozenset]
) -> float:
    r"""Compute the cluster comparison measure.

    The cluster comparison measure is defined as the harmonic mean between
    cluster precision (cp) and cluster recall (cr).

    .. math::

        2 \cdot cp \cdot cr \over cp + cr

    :param ground_truth: a set of sets. Represents the ideal algebraic partition
        induced by the entity resolution relation over an algebraic set.
    :param result: a set of sets. Represents the partition produced by the
        entity resolution task over the same algebraic set as the ground truth.
    """
    cp = cluster_precision(ground_truth, result)
    cr = cluster_recall(ground_truth, result)
    return _safe_division(2 * cp * cr, cp + cr)
