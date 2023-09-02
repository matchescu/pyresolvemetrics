import math
from typing import Callable, Any, Iterable, Generator

from matchescu.adt.types import Record


def _extract_tuple(record: Iterable) -> tuple:
    return tuple(map(_extract_tuple, record)) if isinstance(record, (tuple, list, set)) else record


def extract_serf_result_model(input_data: Iterable[Iterable[Iterable]]) -> list[tuple]:
    model = list(
        tuple(_extract_tuple(record) for record in result)
        for result in input_data
    )
    return model


def _flatten_record_values(record: Record) -> Generator[Any, None, None]:
    for value in record:
        if isinstance(value, (list, set, dict, tuple)):
            yield from _flatten_record_values(value)
        else:
            yield value


def gmd_slice(result: Iterable[Record], standard: Iterable[Record], f_split: Callable[[int, int], float],
              f_merge: Callable[[int, int], float]) -> float:
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
    :param f_split: a cost function of the form
    ``(x: Number, y: Number) -> Number`` that specifies the cost of a **split**
    operation that leads from ``result`` to ``standard``
    :param f_merge: a cost function of the form
    ``(x: Number, y: Number) -> Number`` that specifies the cost of a **merge**
    operation that leads from ``result`` to ``standard``
    :return: a total cost (distance) judged by how many splits and merges
    (and the cost of each of those operations, as specified by the appropriate
    params) must be performed to get from ``result`` to ``standard``.
    """
    res_map = {}
    res_sizes = {}
    for idx, record in enumerate(result):
        for cluster in record:
            res_map[cluster] = idx
        res_sizes[idx] = res_sizes.get(idx, 0) + len(record)

    total_cost = 0

    for record in standard:
        overlap_map = {}
        for cluster in record:
            if cluster not in res_map:
                continue
            record_index = res_map[cluster]
            if record_index not in overlap_map:
                overlap_map[record_index] = 0
            overlap_map[record_index] += 1

        record_cost = 0
        total_overlapping_items = 0
        for i, count in overlap_map.items():
            # count a split if the number of items in the k-th cluster in the result is larger than the overlap
            if res_sizes[i] > count:
                record_cost += f_split(
                    count, res_sizes[i] - count
                )

            # remove the overlap from the k-th cluster in the result
            res_sizes[i] -= count

            if total_overlapping_items != 0:
                # the overlap_map always has only one element when result is included in standard
                # all tokens in the result cluster will be in the same standard cluster
                record_cost += f_merge(
                    count, total_overlapping_items
                )
            total_overlapping_items += count
        total_cost += record_cost

    return total_cost


def basic_merge_distance(result: list[tuple], standard: list[tuple]) -> float:
    return gmd_slice(result, standard, lambda x, y: 1, lambda x, y: 1)


def _compute_partition_of_single_records(records: Iterable[Record]) -> Iterable[Any]:
    ordered_set = {
        item: None
        for record in records
        for item in record
    }
    return [(item,) for item in ordered_set]


def _product(x, y):
    return x * y


def _zero(*_):
    return 0


def pairwise_precision(result: list[tuple], standard: list[tuple]) -> float:
    gt_denormalized = _compute_partition_of_single_records(standard)
    bmd = gmd_slice(result, standard, _product, _zero)
    denorm_precision = gmd_slice(result, gt_denormalized, _product, _zero)

    if denorm_precision == 0:
        return 0.0

    return 1 - (bmd / denorm_precision)


def pairwise_recall(result: list[tuple], standard: list[tuple]) -> float:
    gt_denormalized = _compute_partition_of_single_records(standard)
    bmd = gmd_slice(result, standard, f_split=_zero, f_merge=_product)
    denorm_recall = gmd_slice(gt_denormalized, standard, f_split=_zero, f_merge=_product)

    if (bmd == 0 and denorm_recall == 0) or denorm_recall == 0:
        return 0.0
    return 1 - (bmd / denorm_recall)


def pairwise_f1(result: list[tuple], standard: list[tuple]) -> float:
    precision = pairwise_precision(result, standard)
    recall = pairwise_recall(result, standard)
    return 2 * (precision * recall) / (precision + recall)


def h(x: float, item_count: int) -> float:
    fraction = x / item_count
    return fraction * math.log(fraction)


def variation_of_information(result: list[tuple], standard: list[tuple]) -> float:
    distinct_standard_items = len(set(standard))

    def vi_cost_function(x: float, y: float) -> float:
        return h(x + y, distinct_standard_items) - h(x, distinct_standard_items) - h(y, distinct_standard_items)

    return gmd_slice(result, standard, vi_cost_function, vi_cost_function)
