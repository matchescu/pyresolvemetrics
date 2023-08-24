import math
from typing import Callable, Any, Iterable, Generator

from matchescu.types import Record


def extract_serf_result_model(input_data: Iterable[Iterable[Iterable]]) -> list[tuple]:
    return list(
        tuple(
            tuple(v for v in record) if isinstance(record, (list, tuple, set, dict)) else record
            for record in result
        )
        for result in input_data
    )


def _flatten_record_values(record: Record) -> Generator[Any, None, None]:
    for value in record:
        if isinstance(value, (list, set, dict, tuple)):
            yield from _flatten_record_values(value)
        else:
            yield value


def gmd_slice(ground_truth: Iterable[Record], result: Iterable[Record], split_cost_func: Callable[[int, int], float],
              merge_cost_func: Callable[[int, int], float]) -> float:
    """Generalized merge distance *slice* algorithm.

    This algorithm is used to compute the edit distance between two lists
    of clusters. It was introduced by Menestrina et al. in Appendix C of
    their `Evaluating Entity Resolution Results paper
    <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ee8e13f3f17a2660331a3a17ba8a7cfb06f9b61d>`_.

    :param result: an iterable over iterables where each of the inner
    iterables represents a cluster of information returned by an entity
    resolution algorithm
    :param ground_truth: an iterable over iterables where each of the inner
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
    res_map = {}
    res_sizes = {}
    for idx, record in enumerate(result):
        size = 0
        for value in _flatten_record_values(record):
            res_map[value] = idx
            size += 1
        res_sizes[idx] = size

    cost = 0

    for record in ground_truth:
        overlap_map = {}
        for token in _flatten_record_values(record):
            if token not in res_map:
                continue
            record_index = res_map[token]
            if record_index not in overlap_map:
                overlap_map[record_index] = 0
            overlap_map[record_index] += 1

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


def basic_merge_distance(ground_truth: list[tuple], result: list[tuple]) -> float:
    return gmd_slice(ground_truth, result, lambda x, y: 1, lambda x, y: 1)


def _compute_partition_of_single_records(records: Iterable[Record]) -> Iterable[Any]:
    ordered_set = {
        record: None
        for record in records
    }
    return [item for item in ordered_set]


def _product(x, y):
    return x * y


def _zero(*_):
    return 0


def pairwise_precision(ground_truth: list[tuple], result: list[tuple]) -> float:
    gt_denormalized = _compute_partition_of_single_records(ground_truth)
    return 1 - (
        gmd_slice(ground_truth, result, _product, _zero)
        /
        gmd_slice(gt_denormalized, result, _product, _zero)
    )


def pairwise_recall(ground_truth: list[tuple], result: list[tuple]) -> float:
    gt_denormalized = _compute_partition_of_single_records(ground_truth)
    return 1 - (
        gmd_slice(ground_truth, result, _zero, _product)
        /
        gmd_slice(ground_truth, gt_denormalized, _zero, _product)
    )


def pairwise_f1(ground_truth: list[tuple], result: list[tuple]) -> float:
    precision = pairwise_precision(ground_truth, result)
    recall = pairwise_recall(ground_truth, result)
    return 2 * (precision * recall) / (precision + recall)


def h(x: float, item_count: int) -> float:
    fraction = x / item_count
    return fraction * math.log(fraction)


def variation_of_information(ground_truth: list[tuple], result: list[tuple]) -> float:
    distinct_standard_items = len(set(ground_truth))

    def vi_cost_function(x: float, y: float) -> float:
        return h(x + y, distinct_standard_items) - h(x, distinct_standard_items) - h(y, distinct_standard_items)

    return gmd_slice(ground_truth, result, vi_cost_function, vi_cost_function)
