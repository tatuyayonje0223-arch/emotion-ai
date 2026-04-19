"""Phase 9 metrics: accuracy, per-class precision/recall/F1, confusion matrix,
and McNemar's test for paired baseline comparison.

Uses sklearn + scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)

from phase9.emotion_mapping import EMOTIONAI_LABELS


@dataclass
class EvalResult:
    name: str
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]   # label -> {precision, recall, f1, support}
    confusion: np.ndarray                     # 10x10 matrix
    preds: list[str]                          # for McNemar comparison
    trues: list[str]                          # same; kept for paired tests

    def summary(self) -> str:
        lines = [f"{self.name}: accuracy={self.accuracy:.3f}  macro_F1={self.macro_f1:.3f}"]
        for label in EMOTIONAI_LABELS:
            if label in self.per_class:
                m = self.per_class[label]
                lines.append(f"  {label:<12s} P={m['precision']:.2f} R={m['recall']:.2f} "
                             f"F1={m['f1']:.2f} n={int(m['support'])}")
        return "\n".join(lines)


def evaluate(
    name: str,
    predictions: Sequence[str],
    ground_truth: Sequence[str],
) -> EvalResult:
    """Compute accuracy, per-class P/R/F1, confusion matrix."""
    assert len(predictions) == len(ground_truth), "prediction/truth length mismatch"

    labels = EMOTIONAI_LABELS
    acc = accuracy_score(ground_truth, predictions)
    p, r, f, sup = precision_recall_fscore_support(
        ground_truth, predictions, labels=labels, zero_division=0,
    )
    macro_f1 = float(f.mean())

    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(sup[i]),
        }

    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    return EvalResult(
        name=name, accuracy=float(acc), macro_f1=macro_f1,
        per_class=per_class, confusion=cm,
        preds=list(predictions), trues=list(ground_truth),
    )


def mcnemar_test(a: EvalResult, b: EvalResult) -> dict[str, float]:
    """McNemar's paired-sample test comparing two classifiers on the same items.

    H0: both classifiers have equal error rates.
    Returns dict with b01 (a-correct, b-wrong), b10 (a-wrong, b-correct), chi2, p_value.

    Uses continuity-corrected version (Edwards 1948) for robustness with small samples.
    """
    from scipy.stats import chi2 as chi2_dist

    assert a.trues == b.trues, "McNemar requires identical ground truth order"
    n = len(a.trues)
    b01 = 0  # a correct, b wrong
    b10 = 0  # a wrong, b correct
    for i in range(n):
        t = a.trues[i]
        ac = (a.preds[i] == t)
        bc = (b.preds[i] == t)
        if ac and not bc:
            b01 += 1
        elif (not ac) and bc:
            b10 += 1

    # Continuity-corrected McNemar chi-square
    if b01 + b10 == 0:
        return {"b01": b01, "b10": b10, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p = 1.0 - chi2_dist.cdf(chi2, df=1)
    return {"b01": float(b01), "b10": float(b10), "chi2": float(chi2), "p_value": float(p)}


def format_confusion(result: EvalResult, labels: list[str] | None = None) -> str:
    """ASCII confusion matrix. Rows = true labels, columns = predicted."""
    if labels is None:
        labels = EMOTIONAI_LABELS
    header = "       " + " ".join(f"{l[:4]:>5s}" for l in labels)
    rows = [header]
    for i, true_lbl in enumerate(labels):
        row = f"{true_lbl[:6]:<6s}" + " " + " ".join(
            f"{result.confusion[i][j]:>5d}" for j in range(len(labels)))
        rows.append(row)
    return "\n".join(rows)


if __name__ == "__main__":
    # Sanity check with synthetic data
    gt = ["FEAR"] * 10 + ["SADNESS"] * 10
    pred_good = ["FEAR"] * 9 + ["SADNESS"] + ["SADNESS"] * 10
    pred_bad = ["FEAR"] * 5 + ["SEEKING"] * 5 + ["SADNESS"] * 5 + ["RAGE"] * 5

    good = evaluate("good_clf", pred_good, gt)
    bad = evaluate("bad_clf", pred_bad, gt)

    print(good.summary())
    print()
    print(bad.summary())
    print()
    mcnemar_result = mcnemar_test(good, bad)
    print(f"McNemar good vs bad: {mcnemar_result}")
    print()
    print("Confusion (bad):")
    print(format_confusion(bad))
