"""Evaluation metrics for token-level injection detection, span redaction, and detector evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np


# ==============================================================================
# Token & Span Level Metrics (Token Head & Redaction)
# ==============================================================================

def compute_token_metrics(
    flat_preds: List[int],
    flat_golds: List[int]
) -> Dict[str, float]:
    """
    Compute token-level precision, recall, and F1 ignoring -100 padding/special tokens.
    Evaluates both binary injection detection (B-INJ/I-INJ vs O) and per-label metrics.
    """
    valid_pairs = [(p, g) for p, g in zip(flat_preds, flat_golds) if g != -100]
    if not valid_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    preds, golds = zip(*valid_pairs)

    # Binary injection metrics: positive = label in {1, 2}
    tp = sum(1 for p, g in zip(preds, golds) if p in (1, 2) and g in (1, 2))
    fp = sum(1 for p, g in zip(preds, golds) if p in (1, 2) and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g in (1, 2))
    tn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 0)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-8)

    # Per-class F1
    per_class = {}
    for label_id, name in [(0, "O"), (1, "B-INJ"), (2, "I-INJ")]:
        c_tp = sum(1 for p, g in zip(preds, golds) if p == label_id and g == label_id)
        c_fp = sum(1 for p, g in zip(preds, golds) if p == label_id and g != label_id)
        c_fn = sum(1 for p, g in zip(preds, golds) if p != label_id and g == label_id)
        c_prec = c_tp / max(c_tp + c_fp, 1)
        c_rec = c_tp / max(c_tp + c_fn, 1)
        c_f1 = (2 * c_prec * c_rec) / max(c_prec + c_rec, 1e-8)
        per_class[f"{name}_f1"] = round(c_f1, 4)

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        **per_class
    }


def compute_span_f1(
    predicted_spans_list: List[List[Tuple[int, int]]],
    gold_spans_list: List[List[Tuple[int, int]]],
    iou_threshold: float = 0.0,
    exact_match: bool = False
) -> Dict[str, float]:
    """
    Compute Span-level Precision, Recall, and F1 using 1-to-1 matching.
    Exact match: predicted (s, e) == gold (s, e).
    Overlap match: intersection / union > iou_threshold (or any character overlap if 0.0).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for preds, golds in zip(predicted_spans_list, gold_spans_list):
        matched_golds = set()
        matched_preds = set()

        for p_idx, (p_start, p_end) in enumerate(preds):
            for g_idx, (g_start, g_end) in enumerate(golds):
                if g_idx in matched_golds:
                    continue
                if exact_match:
                    if p_start == g_start and p_end == g_end:
                        matched_golds.add(g_idx)
                        matched_preds.add(p_idx)
                        break
                else:
                    inter = max(0, min(p_end, g_end) - max(p_start, g_start))
                    union = max(p_end, g_end) - min(p_start, g_start)
                    iou = inter / max(union, 1)
                    if (iou_threshold == 0.0 and inter > 0) or (iou > iou_threshold):
                        matched_golds.add(g_idx)
                        matched_preds.add(p_idx)
                        break

        total_tp += len(matched_preds)
        total_fp += len(preds) - len(matched_preds)
        total_fn += len(golds) - len(matched_golds)

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-8)

    prefix = "exact" if exact_match else "overlap"
    return {
        f"span_{prefix}_precision": round(prec, 4),
        f"span_{prefix}_recall": round(rec, 4),
        f"span_{prefix}_f1": round(f1, 4)
    }


def compute_character_coverage(
    document: str,
    gold_spans: List[Tuple[int, int]],
    removed_spans: List[Tuple[int, int]],
    is_blocked: bool
) -> Dict[str, float]:
    """
    Exact character-level security and utility metrics:
    - gold_coverage: fraction of injected characters excised by removed spans.
    - residual_payload: fraction of injected characters surviving in document.
    - legitimate_retained: fraction of non-injected characters surviving in document.
    """
    doc_len = len(document)
    if is_blocked:
        return {
            "gold_coverage": 1.0,
            "residual_payload": 0.0,
            "legitimate_retained": 0.0
        }

    inj_mask = [False] * doc_len
    for s, e in gold_spans:
        for ci in range(s, e):
            inj_mask[ci] = True
    legit_mask = [not m for m in inj_mask]

    removed_mask = [False] * doc_len
    for s, e in removed_spans:
        for ci in range(max(0, s), min(doc_len, e)):
            removed_mask[ci] = True
    surviving_mask = [not m for m in removed_mask]

    total_inj = sum(inj_mask)
    if total_inj > 0:
        surviving_inj = sum(1 for ci in range(doc_len) if inj_mask[ci] and surviving_mask[ci])
        res_payload = surviving_inj / total_inj
        gold_coverage = 1.0 - res_payload
    else:
        res_payload = 0.0
        gold_coverage = 1.0

    total_legit = sum(legit_mask)
    if total_legit > 0:
        surviving_legit = sum(1 for ci in range(doc_len) if legit_mask[ci] and surviving_mask[ci])
        legit_retained = surviving_legit / total_legit
    else:
        legit_retained = 1.0

    return {
        "gold_coverage": round(gold_coverage, 4),
        "residual_payload": round(res_payload, 4),
        "legitimate_retained": round(legit_retained, 4)
    }


def compute_redaction_quality(
    document: str,
    gold_spans: List[Tuple[int, int]],
    redacted_document: str,
    is_blocked: bool
) -> Dict[str, float]:
    """
    Measure redaction quality:
    - residual_payload_rate: Fraction of injected characters still present in redacted text.
    - document_retained: Fraction of original document preserved.
    """
    doc_len = max(len(document), 1)

    if is_blocked:
        return {
            "residual_payload_rate": 0.0,
            "document_retained": 0.0
        }

    # Document retained
    retained_ratio = len(redacted_document) / doc_len

    # Residual payload rate
    total_inj_chars = 0
    residual_chars = 0

    for s, e in gold_spans:
        inj_text = document[s:e]
        total_inj_chars += len(inj_text)
        if inj_text in redacted_document:
            residual_chars += len(inj_text)
        else:
            # Check partial fragment residual
            # Sample 4-gram sliding check
            fragment_len = min(20, len(inj_text))
            if inj_text[:fragment_len] in redacted_document or inj_text[-fragment_len:] in redacted_document:
                residual_chars += len(inj_text) // 2

    res_rate = residual_chars / max(total_inj_chars, 1) if total_inj_chars > 0 else 0.0

    return {
        "residual_payload_rate": round(res_rate, 4),
        "document_retained": round(retained_ratio, 4)
    }


# ==============================================================================
# Document & Detector Level Metrics (Baseline Evaluation)
# ==============================================================================

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


def threshold_at_fpr(
    probabilities: list[float],
    labels: list[int],
    target_fpr: float,
) -> dict[str, Any]:
    """Lowest threshold whose false-positive rate is at most ``target_fpr``.

    A score greater than or equal to the threshold counts as a detection.
    Predicting nothing has false-positive rate 0, so a feasible threshold
    exists whenever the split has a negative. ``tight`` is true when the
    next-lower candidate exceeds the cap.
    """
    if not 0 <= target_fpr <= 1:
        raise ValueError("target FPR must be in [0, 1]")
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        return {
            "threshold": None,
            "reason": "no rows",
            "fpr": None,
            "recall": None,
            "tight": None,
            "fp": 0,
            "negatives": 0,
            "tp": 0,
            "positives": 0,
            "fpr_ci": None,
            "recall_ci": None,
        }

    scores = [float(probability) for probability in probabilities]
    gold = [int(label) for label in labels]
    for label in gold:
        if label not in (0, 1):
            raise ValueError("labels and predictions must be 0 or 1")

    neg_at: dict[float, int] = defaultdict(int)
    pos_at: dict[float, int] = defaultdict(int)
    n_pos = 0
    n_neg = 0
    for score, label in zip(scores, gold, strict=True):
        if label == 1:
            pos_at[score] += 1
            n_pos += 1
        else:
            neg_at[score] += 1
            n_neg += 1

    unique = sorted(set(scores))
    if n_neg == 0:
        threshold = unique[0]
        return {
            "threshold": threshold,
            "reason": "no negatives",
            "fpr": None,
            "recall": None if n_pos == 0 else 1.0,
            "tight": None,
            "fp": 0,
            "negatives": 0,
            "tp": n_pos,
            "positives": n_pos,
            "fpr_ci": None,
            "recall_ci": _wilson_interval(n_pos, n_pos),
        }

    # At threshold t, a row is positive when its score is >= t.
    # Walking unique scores from high to low accumulates that tail.
    neg_ge: dict[float, int] = {}
    pos_ge: dict[float, int] = {}
    fp = 0
    tp = 0
    for score in reversed(unique):
        fp += neg_at[score]
        tp += pos_at[score]
        neg_ge[score] = fp
        pos_ge[score] = tp

    def _at(threshold: float) -> tuple[int, int]:
        if threshold not in neg_ge:
            return 0, 0
        return neg_ge[threshold], pos_ge[threshold]

    candidates = list(unique)
    candidates.append(unique[-1] + 1.0)
    previous: dict[str, float] | None = None
    for threshold in candidates:
        fp, tp = _at(threshold)
        fpr = fp / n_neg
        recall = None if n_pos == 0 else tp / n_pos
        if fpr <= target_fpr:
            return {
                "threshold": threshold,
                "reason": None,
                "fpr": fpr,
                "recall": recall,
                "tight": previous is not None and previous["fpr"] > target_fpr,
                "fp": fp,
                "negatives": n_neg,
                "tp": tp,
                "positives": n_pos,
                "fpr_ci": _wilson_interval(fp, n_neg),
                "recall_ci": _wilson_interval(tp, n_pos),
            }
        previous = {"threshold": threshold, "fpr": float(fpr)}
    raise RuntimeError("predicting nothing must meet every false-positive cap")


def _wilson_interval(successes: int, total: int) -> list[float]:
    from sudarshana.evaluation.p5_common import wilson_ci

    low, high = wilson_ci(int(successes), int(total))
    return [low, high]


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
