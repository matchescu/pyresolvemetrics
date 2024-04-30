from typing import Iterable


def extract_fsm_result_model(input_data: Iterable[Iterable[Iterable]]) -> set[tuple[tuple]]:
    return set(
        tuple(
            tuple(v for v in record)
            for record in result
        )
        for result in input_data
    )


def _true_positives(ground_truth: set[tuple[tuple]], er_result: set[tuple[tuple]]) -> int:
    return sum(1 for match_candidate in er_result if match_candidate in ground_truth)


def _false_positives(ground_truth: set[tuple[tuple]], er_result: set[tuple[tuple]]) -> int:
    return sum(1 for match_candidate in er_result if match_candidate not in ground_truth)


def _false_negatives(ground_truth: set[tuple[tuple]], er_result: set[tuple[tuple]]) -> int:
    return sum(1 for true_match in ground_truth if true_match not in er_result)


def precision(ground_truth: set[tuple[tuple]], er_result: set[tuple[tuple]]) -> float:
    tp = _true_positives(ground_truth, er_result)
    fp = _false_positives(ground_truth, er_result)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(ground_truth: set[tuple[tuple]], er_result: set[tuple[tuple]]) -> float:
    tp = _true_positives(ground_truth, er_result)
    fn = _false_negatives(ground_truth, er_result)
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def f1(ground_truth: set[tuple[tuple]], er_result: set[tuple[tuple]]) -> float:
    p = precision(ground_truth, er_result)
    r = recall(ground_truth, er_result)
    if p + r == 0:
        return 0.0
    return (2*p*r)/(p+r)
