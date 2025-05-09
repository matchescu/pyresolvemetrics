from typing import Hashable

from pyresolvemetrics._utils import _safe_division, _symmetric_hash

Pair = tuple[Hashable, Hashable]


def _true_positives(ground_truth: set[int], result: set[int]) -> int:
    return sum(1 for x in result if x in ground_truth)


def _false_positives(ground_truth: set[int], result: set[int]) -> int:
    return sum(1 for x in result if x not in ground_truth)


def _false_negatives(ground_truth: set[int], result: set[int]) -> int:
    return sum(1 for x in ground_truth if x not in result)


def precision(ground_truth: set[Pair], result: set[Pair]) -> float:
    r"""Evaluate the precision of an entity matching task.

    True positives (tp) are matches in the ``result`` which are also found in the
    ``ground_truth``. False positives (fp) are matches in the ``result`` which are
    not found in the ``ground_truth``. Precision is given by the formula

    .. math::

        tp \over tp + fp

    :param ground_truth: a set of pairs of ``Hashable`` items. Each pair
        represents a known pair of matching entity reference identifiers.
    :param result: a set of pairs of ``Hashable`` items. Each pair represents
        a pair of identifiers for entity references which were output by an
        entity matcher.
    """
    gt_hashes = set(map(_symmetric_hash, ground_truth))
    result_hashes = set(map(_symmetric_hash, result))
    tp = _true_positives(gt_hashes, result_hashes)
    fp = _false_positives(gt_hashes, result_hashes)
    return _safe_division(tp, tp + fp)


def recall(ground_truth: set[Pair], result: set[Pair]) -> float:
    r"""Evaluate the recall of an entity matching task.

    True positives (tp) are matches in the ``result`` which are also found in the
    ``ground_truth``. False negatives (fn) are known good pairs from the
    ``ground_truth`` which are not found in the ``result``.
    Recall is given by the formula

    .. math::

        tp \over tp + fn

    :param ground_truth: a set of pairs of ``Hashable`` items. Each pair
        represents a known pair of matching entity reference identifiers.
    :param result: a set of pairs of ``Hashable`` items. Each pair represents
        a pair of identifiers for entity references which were output by an
        entity matcher.
    """
    gt_hashes = set(map(_symmetric_hash, ground_truth))
    result_hashes = set(map(_symmetric_hash, result))
    tp = _true_positives(gt_hashes, result_hashes)
    fn = _false_negatives(gt_hashes, result_hashes)
    return _safe_division(tp, tp + fn)


def f1(ground_truth: set[Pair], result: set[Pair]) -> float:
    r"""Evaluate the F1 score of an entity matching task.

    The F1 score is computed as the harmonic mean of precision (p) and recall (r).

    .. math::

        2 \cdot p \cdot r \over p + r

    :param ground_truth: a set of pairs of ``Hashable`` items. Each pair
        represents a known pair of matching entity reference identifiers.
    :param result: a set of pairs of ``Hashable`` items. Each pair represents
        a pair of identifiers for entity references which were output by an
        entity matcher.
    """
    p = precision(ground_truth, result)
    r = recall(ground_truth, result)
    return _safe_division(2 * p * r, p + r)
