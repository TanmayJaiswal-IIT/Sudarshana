"""Freeze a detection threshold at a target validation false-positive rate.

Fit on train. Choose the threshold on primary validation rows only. Test,
hard negatives, and the obfuscated held-out set are scored after the
threshold is frozen.

    python -m sudarshana.evaluation.calibrate --config configs/calibration.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from sudarshana.data.io import iter_jsonl, read_jsonl
from sudarshana.data.obfuscation import CLEAN_FAMILY, canonical_sha256
from sudarshana.evaluation.baseline import TruncatedLexicalBaseline, _git_head
from sudarshana.evaluation.fixed_recall import pair_text
from sudarshana.evaluation.metrics import (
    predictions_at_threshold,
    summarize,
    threshold_at_fpr,
    threshold_at_recall,
)

REDACTION_THRESHOLD_NOTE = (
    "Token-span cutoff from configs/prototype_config.yaml. "
    "Not overwritten by the detection false-positive threshold."
)
RECALL_REFERENCE = 0.95


def is_primary_row(row: dict[str, Any]) -> bool:
    """Rows with no id are primary. Counterfactual and obfuscated ids contain ``::``."""
    row_id = row.get("id")
    if row_id is None or str(row_id) == "":
        return True
    return "::" not in str(row_id)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    return (repo_root / path).resolve()


def _repo_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd()


def _view(row: dict[str, Any], mode: str) -> str:
    if mode == "document":
        return str(row["document"])
    if mode == "pair":
        return pair_text(row)
    raise ValueError("mode must be 'document' or 'pair'")


def _round_rate(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _wilson(successes: int, total: int) -> list[float]:
    from sudarshana.evaluation.p5_common import wilson_ci

    low, high = wilson_ci(int(successes), int(total))
    return [low, high]


SCORE_NAME = (
    "truncated lexical log-odds, before the sigmoid. "
    "The sigmoid maps large positive log-odds to exactly 1 and collapses distinct documents."
)


def _scores(model: TruncatedLexicalBaseline, rows: list[dict[str, Any]], mode: str) -> list[float]:
    return [model.score(_view(row, mode)) for row in rows]


def operating_point(
    probabilities: list[float],
    labels: list[int],
    threshold: float,
) -> dict[str, Any]:
    """Recall and false-positive rate at one frozen threshold."""
    summary = summarize(labels, predictions_at_threshold(probabilities, threshold))
    return {
        "threshold": threshold,
        "recall": _round_rate(summary["recall"]),
        "fpr": _round_rate(summary["fpr"]),
        "fp": summary["fp"],
        "negatives": summary["negatives"],
        "tp": summary["tp"],
        "positives": summary["positives"],
        "fpr_ci": _wilson(summary["fp"], summary["negatives"]),
        "recall_ci": _wilson(summary["tp"], summary["positives"]),
    }


def fit_lexical(
    rows: list[dict[str, Any]],
    mode: str,
    max_tokens: int,
) -> tuple[TruncatedLexicalBaseline, int]:
    """Fit on primary rows. Returns the model and the number of rows it used."""
    primary = [row for row in rows if is_primary_row(row)]
    model = TruncatedLexicalBaseline(max_tokens=max_tokens).fit(
        [_view(row, mode) for row in primary],
        [int(row["label"]) for row in primary],
    )
    return model, len(primary)


def _freeze_scored(
    probabilities: list[float],
    labels: list[int],
    *,
    n_excluded: int,
    target_fpr: float,
    target_recall: float,
) -> dict[str, Any]:
    """Choose the false-positive threshold from scores alone."""
    if not probabilities:
        return {
            "threshold": None,
            "reason": "no primary validation rows",
            "n_primary": 0,
            "n_excluded": n_excluded,
            "recall_reference": None,
        }
    chosen = threshold_at_fpr(probabilities, labels, target_fpr)
    chosen["fpr"] = _round_rate(chosen["fpr"])
    chosen["recall"] = _round_rate(chosen["recall"])
    chosen["n_primary"] = len(probabilities)
    chosen["n_excluded"] = n_excluded
    recall_threshold = threshold_at_recall(probabilities, labels, target_recall)
    if recall_threshold is None or chosen["threshold"] is None:
        chosen["recall_reference"] = None
    else:
        reference = operating_point(probabilities, labels, recall_threshold)
        reference["target_recall"] = target_recall
        chosen["recall_reference"] = reference
    return chosen


def freeze_threshold(
    model: TruncatedLexicalBaseline,
    val_rows: list[dict[str, Any]],
    mode: str,
    target_fpr: float,
    *,
    target_recall: float = RECALL_REFERENCE,
) -> dict[str, Any]:
    """Choose the false-positive threshold from validation scores alone."""
    primary = [row for row in val_rows if is_primary_row(row)]
    return _freeze_scored(
        _scores(model, primary, mode),
        [int(row["label"]) for row in primary],
        n_excluded=len(val_rows) - len(primary),
        target_fpr=target_fpr,
        target_recall=target_recall,
    )


def _point(
    probabilities: list[float],
    labels: list[int],
    threshold: float | None,
    rows: int,
) -> dict[str, Any]:
    if threshold is None or rows == 0:
        return {"rows": rows, "threshold": threshold, "recall": None, "fpr": None}
    point = operating_point(probabilities, labels, threshold)
    point["rows"] = rows
    return point


def score_rows(
    model: TruncatedLexicalBaseline,
    rows: list[dict[str, Any]],
    mode: str,
    threshold: float | None,
) -> dict[str, Any]:
    """Apply a frozen threshold. Does not refit it."""
    if threshold is None or not rows:
        return {"rows": len(rows), "threshold": threshold, "recall": None, "fpr": None}
    return _point(_scores(model, rows, mode), [int(row["label"]) for row in rows], threshold, len(rows))


def obfuscated_deltas(
    model: TruncatedLexicalBaseline,
    mode: str,
    threshold: float | None,
    rows: list[dict[str, Any]],
    parents_by_hash: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Child recall minus parent recall for each obfuscation family."""
    if threshold is None or not rows:
        return {"status": "SKIPPED", "reason": "no threshold or no obfuscated rows", "by_family": {}}
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    missing = 0
    for row in rows:
        if int(row.get("label", 0)) != 1:
            continue
        digest = row.get("parent_sha256")
        parent = parents_by_hash.get(digest) if isinstance(digest, str) else None
        if parent is None:
            missing += 1
            continue
        family = str(row.get("obfuscation_family") or "unspecified")
        grouped.setdefault(family, []).append((parent, row))

    by_family: dict[str, Any] = {}
    for family, pairs in grouped.items():
        parent_scores = _scores(model, [parent for parent, _ in pairs], mode)
        child_scores = _scores(model, [child for _, child in pairs], mode)
        parent_hits = sum(1 for score in parent_scores if score >= threshold)
        child_hits = sum(1 for score in child_scores if score >= threshold)
        count = len(pairs)
        by_family[family] = {
            "n": count,
            "parent_hits": parent_hits,
            "child_hits": child_hits,
            "parent_recall": round(parent_hits / count, 4),
            "child_recall": round(child_hits / count, 4),
            "recall_delta": round((child_hits - parent_hits) / count, 4),
            "parent_ci": _wilson(parent_hits, count),
            "child_ci": _wilson(child_hits, count),
        }
    clean = [row for row in rows if row.get("obfuscation_family") == CLEAN_FAMILY or int(row.get("label", 0)) == 0]
    return {
        "status": "OK",
        "missing_parents": missing,
        "by_family": by_family,
        "clean": score_rows(model, clean, mode, threshold),
    }


def redaction_threshold(repo_root: Path) -> float:
    """Read the token-span cutoff. This calibration does not change it."""
    path = repo_root / "configs" / "prototype_config.yaml"
    if not path.is_file():
        return 0.35
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return float(config.get("redaction", {}).get("calibrated_threshold", 0.35))


def load_splits(splits_dir: Path, obfuscated_path: Path | None) -> dict[str, list[dict[str, Any]]]:
    def _read(path: Path) -> list[dict[str, Any]]:
        return read_jsonl(path) if path.is_file() else []

    obfuscated = obfuscated_path if obfuscated_path is not None else splits_dir / "obfuscated.jsonl"
    test = _read(splits_dir / "test.jsonl")
    hard_path = splits_dir / "hard_negatives.jsonl"
    if hard_path.is_file() and hard_path.stat().st_size > 0:
        hard_negatives = _read(hard_path)
    else:
        hard_negatives = [
            row
            for row in test
            if is_primary_row(row)
            and (row.get("source") == "hard_negative" or row.get("attack_category") == "hard_negative")
        ]
    return {
        "train": _read(splits_dir / "train.jsonl"),
        "val": _read(splits_dir / "val.jsonl"),
        "test": test,
        "hard_negatives": hard_negatives,
        "obfuscated": _read(obfuscated),
    }


def parent_index(test_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {canonical_sha256(row): row for row in test_rows}


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _hard_negative_file(splits_dir: Path) -> Path | None:
    path = splits_dir / "hard_negatives.jsonl"
    if path.is_file() and path.stat().st_size > 0:
        return path
    return None


def _is_hard_negative(row: dict[str, Any]) -> bool:
    return row.get("source") == "hard_negative" or row.get("attack_category") == "hard_negative"


def _fit_path(path: Path, mode: str, max_tokens: int) -> tuple[TruncatedLexicalBaseline, int]:
    counted = {"n": 0}

    def pairs():
        for row in iter_jsonl(path):
            if not is_primary_row(row):
                continue
            counted["n"] += 1
            yield _view(row, mode), int(row["label"])

    model = TruncatedLexicalBaseline(max_tokens=max_tokens).fit_pairs(pairs())
    return model, counted["n"]


def _primary_scores(
    model: TruncatedLexicalBaseline,
    path: Path,
    mode: str,
) -> tuple[list[float], list[int], int]:
    scores: list[float] = []
    labels: list[int] = []
    seen = 0
    if not path.is_file():
        return scores, labels, seen
    for row in iter_jsonl(path):
        seen += 1
        if not is_primary_row(row):
            continue
        scores.append(model.score(_view(row, mode)))
        labels.append(int(row["label"]))
    return scores, labels, seen


def _score_file(
    model: TruncatedLexicalBaseline,
    path: Path,
    mode: str,
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    if not path.is_file():
        return scores, labels
    for row in iter_jsonl(path):
        scores.append(model.score(_view(row, mode)))
        labels.append(int(row["label"]))
    return scores, labels


def _score_test(
    model: TruncatedLexicalBaseline,
    path: Path,
    mode: str,
    *,
    derived_hard: bool,
) -> tuple[list[float], list[int], dict[str, float], list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    parents: dict[str, float] = {}
    hard_scores: list[float] = []
    hard_labels: list[int] = []
    if not path.is_file():
        return scores, labels, parents, hard_scores, hard_labels
    for row in iter_jsonl(path):
        if not is_primary_row(row):
            continue
        score = model.score(_view(row, mode))
        label = int(row["label"])
        scores.append(score)
        labels.append(label)
        parents[canonical_sha256(row)] = score
        if derived_hard and _is_hard_negative(row):
            hard_scores.append(score)
            hard_labels.append(label)
    return scores, labels, parents, hard_scores, hard_labels


def _obfuscated_from_path(
    model: TruncatedLexicalBaseline,
    mode: str,
    threshold: float | None,
    path: Path,
    parent_scores: dict[str, float],
) -> dict[str, Any]:
    """Same paired deltas as :func:`obfuscated_deltas`, reading one child at a time."""
    if threshold is None or not path.is_file():
        return {"status": "SKIPPED", "reason": "no threshold or no obfuscated rows", "by_family": {}}
    buckets: dict[str, dict[str, int]] = {}
    missing = 0
    seen = 0
    clean_scores: list[float] = []
    clean_labels: list[int] = []
    for row in iter_jsonl(path):
        seen += 1
        label = int(row.get("label", 0))
        family = row.get("obfuscation_family")
        if label == 1:
            digest = row.get("parent_sha256")
            parent_score = parent_scores.get(digest) if isinstance(digest, str) else None
            if parent_score is None:
                missing += 1
            else:
                name = str(family or "unspecified")
                bucket = buckets.setdefault(name, {"n": 0, "parent_hits": 0, "child_hits": 0})
                bucket["n"] += 1
                if parent_score >= threshold:
                    bucket["parent_hits"] += 1
                if model.score(_view(row, mode)) >= threshold:
                    bucket["child_hits"] += 1
        if family == CLEAN_FAMILY or label == 0:
            clean_scores.append(model.score(_view(row, mode)))
            clean_labels.append(label)
    if seen == 0:
        return {"status": "SKIPPED", "reason": "no threshold or no obfuscated rows", "by_family": {}}
    by_family: dict[str, Any] = {}
    for name, bucket in buckets.items():
        count = bucket["n"]
        by_family[name] = {
            "n": count,
            "parent_hits": bucket["parent_hits"],
            "child_hits": bucket["child_hits"],
            "parent_recall": round(bucket["parent_hits"] / count, 4),
            "child_recall": round(bucket["child_hits"] / count, 4),
            "recall_delta": round((bucket["child_hits"] - bucket["parent_hits"]) / count, 4),
            "parent_ci": _wilson(bucket["parent_hits"], count),
            "child_ci": _wilson(bucket["child_hits"], count),
        }
    return {
        "status": "OK",
        "missing_parents": missing,
        "by_family": by_family,
        "clean": _point(clean_scores, clean_labels, threshold, len(clean_scores)),
    }


def _arm_payload(
    model: TruncatedLexicalBaseline,
    mode: str,
    *,
    n_train: int,
    max_tokens: int,
    target_fpr: float,
    frozen: dict[str, Any],
    test: dict[str, Any],
    hard: dict[str, Any],
    obfuscated: dict[str, Any],
) -> dict[str, Any]:
    threshold = frozen.get("threshold")
    return {
        "model": "truncated-lexical",
        "score": SCORE_NAME,
        "input": "document" if mode == "document" else "user_task_then_document",
        "uses_user_task": mode == "pair",
        "max_tokens": max_tokens,
        "n_train": n_train,
        "target_fpr": target_fpr,
        "threshold": threshold,
        "tight": frozen.get("tight"),
        "reason": frozen.get("reason"),
        "n_primary": frozen.get("n_primary"),
        "n_excluded": frozen.get("n_excluded"),
        "val": {
            "threshold": threshold,
            "recall": frozen.get("recall"),
            "fpr": frozen.get("fpr"),
            "fp": frozen.get("fp"),
            "negatives": frozen.get("negatives"),
            "tp": frozen.get("tp"),
            "positives": frozen.get("positives"),
            "fpr_ci": frozen.get("fpr_ci"),
            "recall_ci": frozen.get("recall_ci"),
            "tight": frozen.get("tight"),
        },
        "recall_reference": frozen.get("recall_reference"),
        "test": test,
        "hard_negatives": hard,
        "obfuscated": obfuscated,
    }


def evaluate_arm(
    train_rows: list[dict[str, Any]],
    eval_rows: dict[str, list[dict[str, Any]]],
    mode: str,
    *,
    target_fpr: float,
    max_tokens: int,
    target_recall: float = RECALL_REFERENCE,
) -> dict[str, Any]:
    """Fit one lexical view, freeze its threshold on validation, then score the rest."""
    model, n_train = fit_lexical(train_rows, mode, max_tokens)
    frozen = freeze_threshold(
        model,
        eval_rows.get("val") or [],
        mode,
        target_fpr,
        target_recall=target_recall,
    )
    threshold = frozen.get("threshold")
    test_rows = [row for row in (eval_rows.get("test") or []) if is_primary_row(row)]
    parents = parent_index(test_rows)
    return _arm_payload(
        model,
        mode,
        n_train=n_train,
        max_tokens=max_tokens,
        target_fpr=target_fpr,
        frozen=frozen,
        test=score_rows(model, test_rows, mode, threshold),
        hard=score_rows(model, eval_rows.get("hard_negatives") or [], mode, threshold),
        obfuscated=obfuscated_deltas(
            model,
            mode,
            threshold,
            eval_rows.get("obfuscated") or [],
            parents,
        ),
    )


def evaluate_arm_from_paths(
    splits_dir: Path,
    obfuscated_path: Path | None,
    mode: str,
    *,
    target_fpr: float,
    max_tokens: int,
    target_recall: float = RECALL_REFERENCE,
) -> dict[str, Any]:
    """Same arm as :func:`evaluate_arm`, streaming each split so the full corpus stays on disk."""
    model, n_train = _fit_path(splits_dir / "train.jsonl", mode, max_tokens)
    val_scores, val_labels, n_val = _primary_scores(model, splits_dir / "val.jsonl", mode)
    frozen = _freeze_scored(
        val_scores,
        val_labels,
        n_excluded=n_val - len(val_scores),
        target_fpr=target_fpr,
        target_recall=target_recall,
    )
    threshold = frozen.get("threshold")
    hard_file = _hard_negative_file(splits_dir)
    test_scores, test_labels, parent_scores, hard_scores, hard_labels = _score_test(
        model,
        splits_dir / "test.jsonl",
        mode,
        derived_hard=hard_file is None,
    )
    if hard_file is not None:
        hard_scores, hard_labels = _score_file(model, hard_file, mode)
    obfuscated = obfuscated_path if obfuscated_path is not None else splits_dir / "obfuscated.jsonl"
    return _arm_payload(
        model,
        mode,
        n_train=n_train,
        max_tokens=max_tokens,
        target_fpr=target_fpr,
        frozen=frozen,
        test=_point(test_scores, test_labels, threshold, len(test_scores)),
        hard=_point(hard_scores, hard_labels, threshold, len(hard_scores)),
        obfuscated=_obfuscated_from_path(model, mode, threshold, obfuscated, parent_scores),
    )


def build_calibration(
    splits: dict[str, list[dict[str, Any]]],
    *,
    target_fpr: float,
    max_tokens: int,
    seed: int,
    target_recall: float,
    repo_root: Path,
) -> dict[str, Any]:
    """Calibrate the document and pair lexical arms. Test labels are not an input to the threshold."""
    shared = {
        "target_fpr": target_fpr,
        "max_tokens": max_tokens,
        "target_recall": target_recall,
    }
    arms = {
        mode: evaluate_arm(splits["train"], splits, mode, **shared)
        for mode in ("document", "pair")
    }
    return {
        "schema_version": "1.1",
        "target_fpr": target_fpr,
        "target_recall_reference": target_recall,
        "seed": seed,
        "max_tokens": max_tokens,
        "calibration_split": "val",
        "redaction_threshold": redaction_threshold(repo_root),
        "redaction_threshold_note": REDACTION_THRESHOLD_NOTE,
        "git_head": _git_head(),
        "row_counts": {name: len(rows) for name, rows in splits.items()},
        "arms": arms,
    }


def _hard_negative_count(splits_dir: Path) -> int:
    hard_file = _hard_negative_file(splits_dir)
    if hard_file is not None:
        return _count_jsonl(hard_file)
    test_path = splits_dir / "test.jsonl"
    if not test_path.is_file():
        return 0
    return sum(1 for row in iter_jsonl(test_path) if is_primary_row(row) and _is_hard_negative(row))


def build_calibration_from_paths(
    splits_dir: Path,
    obfuscated_path: Path | None,
    *,
    target_fpr: float,
    max_tokens: int,
    seed: int,
    target_recall: float,
    repo_root: Path,
) -> dict[str, Any]:
    """Calibrate from JSONL paths. Numerically the same as loading the files first."""
    obfuscated = obfuscated_path if obfuscated_path is not None else splits_dir / "obfuscated.jsonl"
    shared = {
        "target_fpr": target_fpr,
        "max_tokens": max_tokens,
        "target_recall": target_recall,
    }
    arms = {
        mode: evaluate_arm_from_paths(splits_dir, obfuscated, mode, **shared)
        for mode in ("document", "pair")
    }
    return {
        "schema_version": "1.1",
        "target_fpr": target_fpr,
        "target_recall_reference": target_recall,
        "seed": seed,
        "max_tokens": max_tokens,
        "calibration_split": "val",
        "redaction_threshold": redaction_threshold(repo_root),
        "redaction_threshold_note": REDACTION_THRESHOLD_NOTE,
        "git_head": _git_head(),
        "row_counts": {
            "train": _count_jsonl(splits_dir / "train.jsonl"),
            "val": _count_jsonl(splits_dir / "val.jsonl"),
            "test": _count_jsonl(splits_dir / "test.jsonl"),
            "hard_negatives": _hard_negative_count(splits_dir),
            "obfuscated": _count_jsonl(obfuscated),
        },
        "arms": arms,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_from_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    splits_dir = _resolve(config_path, str(config.get("splits_dir", "data/splits")))
    obfuscated = _resolve(config_path, str(config["obfuscated"])) if config.get("obfuscated") else None
    recorded = build_calibration_from_paths(
        splits_dir,
        obfuscated,
        target_fpr=float(config.get("target_fpr", 0.05)),
        max_tokens=int(config.get("max_tokens", 512)),
        seed=int(config.get("seed", 2023)),
        target_recall=float(config.get("target_recall_reference", RECALL_REFERENCE)),
        repo_root=_repo_root(config_path),
    )
    output = _resolve(config_path, str(config.get("output", "results/calibration_day6.json")))
    write_json(output, recorded)
    document = recorded["arms"]["document"]
    print(
        "calibration target_fpr={target} document threshold={threshold} val_fpr={fpr} -> {path}".format(
            target=recorded["target_fpr"],
            threshold=document.get("threshold"),
            fpr=document.get("val", {}).get("fpr"),
            path=output,
        )
    )
    return recorded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate a detection threshold at a target FPR.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_from_config(config, config_path)


if __name__ == "__main__":
    main()
