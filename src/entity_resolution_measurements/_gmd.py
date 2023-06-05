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
    result_map, result_sizes = _build_cluster_map(result)
    cost = 0

    for cluster in standard:
        result_exists = {}
        for token in cluster:
            cluster_index = result_map[token]
            if cluster_index not in result_exists:
                result_exists[cluster_index] = 0
            result_exists[cluster_index] += 1
        si_cost = 0
        total_recs = 0
        for cluster_index, standard_item_count in result_exists.items():
            if result_sizes[cluster_index] > standard_item_count:  # count a split
                si_cost += split_cost_func(
                    standard_item_count, result_sizes[cluster_index] - standard_item_count
                )
            result_sizes[cluster_index] -= standard_item_count
            if total_recs != 0:  # has to merge
                si_cost += merge_cost_func(
                    standard_item_count, total_recs
                )
            total_recs += standard_item_count
        cost += si_cost

    return cost


def basic_merge_distance(result: Clustering, standard: Clustering) -> float:
    common_length = min(len(result), len(standard))
    total_score = 0
    for i in range(common_length):
        pairwise_bmd = _slice(
            result.clustered_rows[i],
            standard.clustered_rows[i],
            lambda x, y: 1,
            lambda x, y: 1
        )
        total_score += pairwise_bmd
    return total_score
