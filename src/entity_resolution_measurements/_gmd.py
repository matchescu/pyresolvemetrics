import math
from typing import Callable, Any, Iterable, Generator

from abstractions.data_structures import Clustering
from abstractions.protocols import SizedIterable


def _build_cluster_map(record: Iterable[SizedIterable]) -> tuple[dict[Any, set[int]], list[int]]:
    cluster_map = {}
    cluster_sizes = []
    for i, cluster in enumerate(record):
        for j, token in enumerate(cluster):
            if token in cluster_map:
                cluster_map[token].add(i)
            else:
                cluster_map[token] = {i}
        cluster_sizes.append(len(cluster))
    return cluster_map, cluster_sizes


def _safe_inc(obj: dict[int, int], key: set[int]) -> dict:
    for idx in key:
        obj[idx] = obj.get(idx, 0) + 1
    return obj


def _build_equivalent_partition(record: Iterable[SizedIterable]) -> Generator[SizedIterable, None, None]:
    union = set()
    for cluster in record:
        cluster_set = set(cluster)
        cluster_unique = cluster_set - union
        union |= cluster_set
        if len(cluster_unique) > 0:
            yield list(cluster_unique)


def gmd_slice(
    result: Iterable[SizedIterable],
    standard: Iterable[SizedIterable],
    split_cost_func: Callable[[int, int], float],
    merge_cost_func: Callable[[int, int], float],
) -> float:
    """Generalized merge distance *slice* algorithm.

    This algorithm is used to compute the edit distance between two lists
    of clusters. It was introduced by Menestrina et al. in Appendix C of
    their `Evaluating Entity Resolution Results paper
    <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ee8e13f3f17a2660331a3a17ba8a7cfb06f9b61d>`_.

    :param result: an iterable over iterables where each of the inner
    iterables represents a cluster of information returned by an entity
    resolution algorithm
    :param standard: an iterable over iterables where each of the inner
    iterables represents a cluster of information from the golden standard
    :param split_cost_func: a cost function of the form
    ``(x: Number, y: Number) -> Number`` that specifies the cost of a **split**
    operation that leads from ``result`` to ``standard``
    :param merge_cost_func: a cost function of the form
    ``(x: Number, y: Number) -> Number`` that specifies the cost of a **merge**
    operation that leads from ``result`` to ``standard``
    :return: a total cost (distance) judged by how many splits and merges
    (and the cost of each of those operations, as specified by the appropriate
    params) must be performed to get from ``result`` to ``standard``.
    """

    result = list(_build_equivalent_partition(result))
    standard = list(_build_equivalent_partition(standard))

    res_map, res_sizes = _build_cluster_map(result)
    cost = 0

    for cluster in standard:
        overlap_map = {}
        for token in cluster:
            try:
                k = res_map[token]
                overlap_map = _safe_inc(overlap_map, k)
            except KeyError:
                raise ValueError(f"the token '{token}' is in the standard, but not in the result")

        standard_cluster_cost = 0
        total_tokens = 0
        for k, overlapped_item_count in overlap_map.items():
            # count a split if the number of items in the k-th cluster in the result is larger than the overlap
            if res_sizes[k] > overlapped_item_count:
                standard_cluster_cost += split_cost_func(
                    overlapped_item_count, res_sizes[k] - overlapped_item_count
                )

            # remove the overlap from the k-th cluster in the result
            res_sizes[k] -= overlapped_item_count

            if total_tokens != 0:
                # the overlap_map always has only one element when result is included in standard
                # all tokens in the result cluster will be in the same standard cluster
                standard_cluster_cost += merge_cost_func(
                    overlapped_item_count, total_tokens
                )
            total_tokens += overlapped_item_count
        cost += standard_cluster_cost

    return cost


def basic_merge_distance(result: Clustering, standard: Clustering) -> float:
    common_length = min(len(result), len(standard))
    cumulated_merge_distance = 0
    for i in range(common_length):
        pairwise_bmd = gmd_slice(
            result.clustered_rows[i],
            standard.clustered_rows[i],
            lambda x, y: 1,
            lambda x, y: 1
        )
        cumulated_merge_distance += pairwise_bmd
    return cumulated_merge_distance


def _denormalize(clusters: Iterable[tuple]) -> Iterable[tuple]:
    return [
        (item,)
        for cluster in clusters
        for item in cluster
    ]


def pairwise_f1(result: Clustering, standard: Clustering) -> list[float]:
    common_length = min(len(result), len(standard))
    ret = []

    def _product(x, y):
        return x*y

    def _zero(*_):
        return 0

    for i in range(common_length):
        resi = result.clustered_rows[i]
        stdi = standard.clustered_rows[i]
        stdi_denormalized = _denormalize(stdi)
        denorm_precision = gmd_slice(resi, stdi_denormalized, _product, _zero)
        denorm_recall = gmd_slice(stdi_denormalized, stdi, _zero, _product)
        if denorm_precision == 0 or denorm_recall == 0:
            ret.append(0)
            continue
        precision = 1 - (gmd_slice(resi, stdi, _product, _zero) / denorm_precision)
        recall = 1 - (gmd_slice(resi, stdi, _zero, _product) / denorm_recall)
        ret.insert(i, 2 * (precision * recall) / (precision + recall))

    return ret


def h(x: float, item_count: int) -> float:
    fraction = x / item_count
    return fraction * math.log(fraction)


def variation_of_information(result: Clustering, standard: Clustering) -> list[float]:
    common_length = min(len(result), len(standard))
    vi = []
    for i in range(common_length):
        distinct_standard_items = len(set(item for cluster in standard.clustered_rows[i] for item in cluster))

        def vi_cost_function(x: float, y: float) -> float:
            return h(x+y, distinct_standard_items) - h(x, distinct_standard_items) - h(y, distinct_standard_items)
        vi.append(
            gmd_slice(result.clustered_rows[i], standard.clustered_rows[i], vi_cost_function, vi_cost_function)
        )
    return vi
