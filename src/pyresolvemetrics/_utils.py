from typing import Hashable


def _safe_division(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0


def _symmetric_hash(pair: tuple[Hashable, Hashable]) -> int:
    return hash(frozenset(pair))
