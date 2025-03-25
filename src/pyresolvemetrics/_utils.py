def _safe_division(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0
