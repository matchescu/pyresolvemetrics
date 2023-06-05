import math
from typing import Callable, Any, Iterable

from abstractions.data_structures import Clustering


def _build_cluster_map(record: Iterable[tuple]) -> tuple[dict[Any, int], list[int]]:
    cluster_map = {}
    cluster_sizes = []
    for i, cluster in enumerate(record):
        for j, token in enumerate(cluster):
            cluster_map[token] = i
        cluster_sizes.append(len(cluster))
    return cluster_map, cluster_sizes


def _slice(
        result: Iterable[tuple],
        standard: Iterable[tuple],
        split_cost_func: Callable[[int, int], float],
        merge_cost_func: Callable[[int, int], float],
) -> float:
    res_map, res_sizes = _build_cluster_map(result)
    cost = 0

    for cluster in standard:
        rs_overlap = {}
        for token in cluster:
            cluster_index = res_map[token]
            if cluster_index not in rs_overlap:
                rs_overlap[cluster_index] = 0
            rs_overlap[cluster_index] += 1
        si_cost = 0
        total_recs = 0
        for cluster_index, standard_item_count in rs_overlap.items():
            if res_sizes[cluster_index] > standard_item_count:  # count a split
                si_cost += split_cost_func(
                    standard_item_count, res_sizes[cluster_index] - standard_item_count
                )
            res_sizes[cluster_index] -= standard_item_count
            if total_recs != 0:  # has to merge
                si_cost += merge_cost_func(
                    standard_item_count, total_recs
                )
            total_recs += standard_item_count
        cost += si_cost

    return cost


def basic_merge_distance(result: Clustering, standard: Clustering) -> float:
    common_length = min(len(result), len(standard))
    cumulated_merge_distance = 0
    for i in range(common_length):
        pairwise_bmd = _slice(
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
        denorm_precision = _slice(resi, stdi_denormalized, _product, _zero)
        denorm_recall = _slice(stdi_denormalized, stdi, _zero, _product)
        if denorm_precision == 0 or denorm_recall == 0:
            ret.append(0)
            continue
        precision = 1 - (_slice(resi, stdi, _product, _zero) / denorm_precision)
        recall = 1 - (_slice(resi, stdi, _zero, _product) / denorm_recall)
        ret.append(2 * (precision * recall) / (precision + recall))

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
            _slice(result.clustered_rows[i], standard.clustered_rows[i], vi_cost_function, vi_cost_function)
        )
    return vi
