"""FPR at a fixed recall for the document-only and pair lexical models.

The threshold is the highest score on primary validation positives whose
recall is at least the target. Primary rows are those whose id has no ``::``.
Counterfactual and mismatched siblings are scored at that same threshold and
reported apart from the hard-negative FPR.

    python -m sudarshana.evaluation.fixed_recall --config configs/fixed_recall.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from sudarshana.data.io import read_jsonl
from sudarshana.evaluation.baseline import TruncatedLexicalBaseline, _git_head, evaluate_predictions
from sudarshana.evaluation.metrics import negative_fpr, predictions_at_threshold, threshold_at_recall

PAIR_NOTE = (
    "Pair text is the user task, a newline, then the document, truncated once "
    "to max_tokens. A long user task can crowd the document out of that window."
)


def pair_text(row: dict[str, Any]) -> str:
    return f"{row['user_task']}\n{row['document']}"


def is_primary(row: dict[str, Any]) -> bool:
    return "::" not in row["id"]


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    return (repo_root / path).resolve()


def _view(row: dict[str, Any], mode: str) -> str:
    if mode == "document":
        return row["document"]
    if mode == "pair":
        return pair_text(row)
    raise ValueError("mode must be 'document' or 'pair'")


def _subset_report(
    rows: list[dict],
    model: TruncatedLexicalBaseline,
    mode: str,
    threshold: float,
) -> dict[str, Any]:
    primary = [row for row in rows if is_primary(row)]
    counterfactual = [row for row in rows if row["id"].endswith("::counterfactual")]
    mismatched = [row for row in rows if row["id"].endswith("::mismatched_task")]

    def predict(subset: list[dict]) -> list[int]:
        if not subset:
            return []
        probabilities = model.predict_proba([_view(row, mode) for row in subset])
        return predictions_at_threshold(probabilities, threshold)

    report = evaluate_predictions(primary, predict(primary)) if primary else {"overall": None, "rows": 0}
    report["rows"] = len(primary)
    report["fpr_counterfactual"] = negative_fpr(
        counterfactual,
        predict(counterfactual),
        lambda row: True,
    )
    report["fpr_mismatched_hard_negative"] = negative_fpr(
        mismatched,
        predict(mismatched),
        lambda row: True,
    )
    report["fpr_mismatched_hard_negative_by_document_source"] = _fpr_by_document_source(
        mismatched,
        predict(mismatched),
    )
    return report


def _fpr_by_document_source(rows: list[dict], predictions: list[int]) -> dict[str, dict]:
    names = sorted({str(row["document_source"]) for row in rows})
    return {
        name: negative_fpr(
            rows,
            predictions,
            lambda row, source=name: str(row.get("document_source")) == source,
        )
        for name in names
    }


def _fit_threshold(
    model: TruncatedLexicalBaseline,
    val_rows: list[dict],
    mode: str,
    target_recall: float,
) -> tuple[float | None, float | None]:
    primary = [row for row in val_rows if is_primary(row)]
    if not primary:
        return None, None
    probabilities = model.predict_proba([_view(row, mode) for row in primary])
    labels = [int(row["label"]) for row in primary]
    threshold = threshold_at_recall(probabilities, labels, target_recall)
    if threshold is None:
        return None, None
    predictions = predictions_at_threshold(probabilities, threshold)
    positives = [prediction for label, prediction in zip(labels, predictions, strict=True) if label == 1]
    if not positives:
        return threshold, None
    return threshold, sum(positives) / len(positives)


def run_fixed_recall(
    train_rows: list[dict],
    eval_rows: dict[str, list[dict]],
    *,
    target_recall: float = 0.95,
    max_tokens: int = 512,
    seed: int = 2023,
) -> dict[str, Any]:
    """Fit both lexical views and calibrate each on primary validation positives."""
    primary_train = [row for row in train_rows if is_primary(row)]
    document_model = TruncatedLexicalBaseline(max_tokens=max_tokens).fit(
        [row["document"] for row in primary_train],
        [int(row["label"]) for row in primary_train],
    )
    pair_model = TruncatedLexicalBaseline(max_tokens=max_tokens).fit(
        [pair_text(row) for row in train_rows],
        [int(row["label"]) for row in train_rows],
    )
    runs = {
        "document": _run_model(
            document_model,
            "document",
            eval_rows,
            target_recall=target_recall,
            n_train=len(primary_train),
        ),
        "pair": _run_model(
            pair_model,
            "pair",
            eval_rows,
            target_recall=target_recall,
            n_train=len(train_rows),
        ),
    }
    status = "lexical_pair+lexical_document" if all(run.get("threshold") is not None for run in runs.values()) else "not_run"
    return {
        "schema_version": "1.1",
        "status": status,
        "target_recall": target_recall,
        "seed": seed,
        "max_tokens": max_tokens,
        "pair_note": PAIR_NOTE,
        "row_counts": {
            "train": len(train_rows),
            "train_primary": len(primary_train),
            "val": len(eval_rows.get("val") or []),
            "test": len(eval_rows.get("test") or []),
        },
        "git_head": _git_head(),
        "runs": runs,
    }


def _run_model(
    model: TruncatedLexicalBaseline,
    mode: str,
    eval_rows: dict[str, list[dict]],
    *,
    target_recall: float,
    n_train: int,
) -> dict[str, Any]:
    threshold, achieved = _fit_threshold(model, eval_rows.get("val") or [], mode, target_recall)
    report: dict[str, Any] = {
        "model": "truncated-lexical",
        "input": "document" if mode == "document" else "user_task_then_document",
        "uses_user_task": mode == "pair",
        "max_tokens": model.max_tokens,
        "target_recall": target_recall,
        "threshold": threshold,
        "achieved_val_recall": achieved,
        "n_train": n_train,
        "splits": {},
    }
    if threshold is None:
        return report
    for name, rows in eval_rows.items():
        report["splits"][name] = _subset_report(rows, model, mode, threshold) if rows else {"overall": None, "rows": 0}
    return report


def run_from_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    data_dir = _resolve(config_path, config.get("data_dir", "data/processed"))
    train_rows = read_jsonl(data_dir / "train.jsonl")
    eval_rows = {
        "val": read_jsonl(data_dir / "val.jsonl") if (data_dir / "val.jsonl").is_file() else [],
        "test": read_jsonl(data_dir / "test.jsonl"),
    }
    recorded = run_fixed_recall(
        train_rows,
        eval_rows,
        target_recall=float(config.get("target_recall", 0.95)),
        max_tokens=int(config.get("max_tokens", 512)),
        seed=int(config.get("seed", 2023)),
    )
    payload = json.dumps(recorded, indent=2)
    if config.get("record"):
        record_path = _resolve(config_path, str(config["record"]))
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(payload, encoding="utf-8")
    output = _resolve(config_path, config.get("output", "data/processed/fixed_recall_metrics.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    test_run = recorded["runs"]["pair"]["splits"].get("test", {}).get("overall")
    print(
        "fixed-recall status={status} pair test recall={recall} hard-negative fpr={fpr} -> {path}".format(
            status=recorded["status"],
            recall=None if not test_run else test_run.get("recall"),
            fpr=None
            if not recorded["runs"]["pair"]["splits"].get("test")
            else recorded["runs"]["pair"]["splits"]["test"].get("fpr_hard_negative", {}).get("fpr"),
            path=output,
        )
    )
    return recorded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record FPR at a fixed recall.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_from_config(config, args.config.resolve())


if __name__ == "__main__":
    main()
