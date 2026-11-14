"""Recall, false-positive rate, and the Day-1 breakdowns."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def binary_counts(labels: Iterable[int], predictions: Iterable[int]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for label, prediction in zip(labels, predictions, strict=True):
        if label not in (0, 1) or prediction not in (0, 1):
            raise ValueError("labels and predictions must be 0 or 1")
        if label == 1 and prediction == 1:
            counts["tp"] += 1
        elif label == 0 and prediction == 1:
            counts["fp"] += 1
        elif label == 0 and prediction == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def summarize(labels: list[int], predictions: list[int]) -> dict[str, float | int | None]:
    counts = binary_counts(labels, predictions)
    return {
        **counts,
        "support": len(labels),
        "positives": counts["tp"] + counts["fn"],
        "negatives": counts["tn"] + counts["fp"],
        "recall": _rate(counts["tp"], counts["tp"] + counts["fn"]),
        "fpr": _rate(counts["fp"], counts["fp"] + counts["tn"]),
        "precision": _rate(counts["tp"], counts["tp"] + counts["fp"]),
    }


def threshold_at_recall(
    probabilities: list[float],
    labels: list[int],
    target: float,
) -> float | None:
    """Highest threshold whose recall on the positives is at least ``target``.

    Scores equal to the threshold count as positive, so a tie can only raise
    recall. No positives returns None.
    """
    if not 0 < target <= 1:
        raise ValueError("target recall must be in (0, 1]")
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    positive = sorted(
        float(probability)
        for probability, label in zip(probabilities, labels, strict=True)
        if int(label) == 1
    )
    if not positive:
        return None
    needed = math.ceil(target * len(positive))
    return positive[len(positive) - needed]


def predictions_at_threshold(probabilities: list[float], threshold: float) -> list[int]:
    return [1 if probability >= threshold else 0 for probability in probabilities]


def negative_fpr(rows: list[dict], predictions: list[int], predicate) -> dict[str, float | int | None]:
    chosen_labels: list[int] = []
    chosen_predictions: list[int] = []
    for row, prediction in zip(rows, predictions, strict=True):
        if int(row["label"]) != 0 or not predicate(row):
            continue
        chosen_labels.append(0)
        chosen_predictions.append(int(prediction))
    if not chosen_labels:
        return {"fpr": None, "support": 0, "fp": 0, "tn": 0}
    summary = summarize(chosen_labels, chosen_predictions)
    return {
        "fpr": summary["fpr"],
        "support": summary["negatives"],
        "fp": summary["fp"],
        "tn": summary["tn"],
    }


def hard_negative_fpr_by_source(rows: list[dict], predictions: list[int]) -> dict[str, dict]:
    names = sorted(
        {
            str(row["document_source"])
            for row in rows
            if int(row["label"]) == 0 and row.get("attack_category") == "hard_negative"
        }
    )
    return {
        name: negative_fpr(
            rows,
            predictions,
            lambda row, source=name: row.get("attack_category") == "hard_negative"
            and str(row.get("document_source")) == source,
        )
        for name in names
    }


def grouped_recall(
    rows: list[dict],
    predictions: list[int],
    key: str,
) -> dict[str, dict]:
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        if row["label"] != 1:
            continue
        group = row.get(key)
        name = "none" if group is None else str(group)
        buckets[name].append((1, prediction))
    report = {}
    for name in sorted(buckets):
        labels = [item[0] for item in buckets[name]]
        preds = [item[1] for item in buckets[name]]
        summary = summarize(labels, preds)
        report[name] = {
            "support": summary["positives"],
            "recall": summary["recall"],
            "tp": summary["tp"],
            "fn": summary["fn"],
        }
    return report
