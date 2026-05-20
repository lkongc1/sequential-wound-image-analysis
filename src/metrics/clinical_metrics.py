"""Clinical metrics: wound area, color changes, healing progression.

Additionally provides F2-score (recall-weighted F-beta) and NPV
(Negative Predictive Value) for FDA 510(k) clinical evaluation.
"""


def f2_score(tp: int, fp: int, fn: int, tn: int, beta: float = 2.0) -> float:
    """Calculate the F-beta score with beta=2.0 (recall-weighted).

    F-beta = (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        tn: True negatives (unused but accepted for call-site consistency).
        beta: Weight parameter; beta > 1 favors recall over precision.

    Returns:
        F-beta score between 0.0 and 1.0, or 0.0 if denominator is zero.
    """
    beta2 = beta * beta
    numerator = (1 + beta2) * tp
    denominator = (1 + beta2) * tp + beta2 * fn + fp
    return numerator / denominator if denominator > 0 else 0.0


def npv(tn: int, fn: int) -> float:
    """Calculate Negative Predictive Value.

    NPV = TN / (TN + FN)

    Args:
        tn: True negatives.
        fn: False negatives.

    Returns:
        NPV between 0.0 and 1.0, or 0.0 if denominator is zero.
    """
    denom = tn + fn
    return tn / denom if denom > 0 else 0.0
