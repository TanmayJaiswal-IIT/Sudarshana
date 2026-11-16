"""Day-1 baseline: one string, truncated at 512 tokens, user task ignored.

This is ablation step 1. The input is ``document`` only. Tokens past the
cutoff are dropped before the classifier sees them, which is the failure mode
later chunking is meant to fix. Pair input is not used here.

The default model is a bag-of-words log-odds classifier so the harness runs
on CPU. ``model: deberta`` trains ``microsoft/deberta-v3-base`` with the same
truncation, when torch and transformers are installed.

    python -m sudarshana.evaluation.baseline --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from sudarshana.data.io import read_jsonl
from sudarshana.evaluation.metrics import grouped_recall, hard_negative_fpr_by_source, negative_fpr, summarize

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def truncate_text(text: str, max_tokens: int) -> str:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def _features(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


class TruncatedLexicalBaseline:
    """Presence log-odds over the first ``max_tokens`` whitespace tokens."""

    def __init__(self, max_tokens: int = 512, threshold: float = 0.5) -> None:
        self.max_tokens = max_tokens
        self.threshold = threshold
        self.weights: dict[str, float] = {}
        self.bias = 0.0

    def fit(self, documents: list[str], labels: list[int]) -> "TruncatedLexicalBaseline":
        if len(documents) != len(labels):
            raise ValueError("documents and labels must have the same length")
        return self.fit_pairs(zip(documents, labels, strict=True))

    def fit_pairs(self, pairs) -> "TruncatedLexicalBaseline":
        """Fit from ``(document, label)`` pairs. One pass, so a file can be streamed."""
        positive = Counter()
        negative = Counter()
        n_pos = 0
        n_neg = 0
        for document, label in pairs:
            features = _features(truncate_text(document, self.max_tokens))
            if label == 1:
                positive.update(features)
                n_pos += 1
            elif label == 0:
                negative.update(features)
                n_neg += 1
            else:
                raise ValueError("labels must be 0 or 1")
        if n_pos == 0 or n_neg == 0:
            raise ValueError("the baseline needs both clean and injected training rows")
        alpha = 0.5
        self.bias = math.log((n_pos + alpha) / (n_neg + alpha))
        weights: dict[str, float] = {}
        for token in set(positive) | set(negative):
            p_pos = (positive[token] + alpha) / (n_pos + 2 * alpha)
            p_neg = (negative[token] + alpha) / (n_neg + 2 * alpha)
            weights[token] = math.log(p_pos) - math.log(p_neg)
        self.weights = weights
        return self

    def score(self, document: str) -> float:
        total = self.bias
        for token in _features(truncate_text(document, self.max_tokens)):
            total += self.weights.get(token, 0.0)
        return total

    def predict_proba(self, documents: list[str]) -> list[float]:
        probabilities = []
        for document in documents:
            value = self.score(document)
            if value >= 0:
                probabilities.append(1.0 / (1.0 + math.exp(-value)))
            else:
                exp_value = math.exp(value)
                probabilities.append(exp_value / (1.0 + exp_value))
        return probabilities

    def predict(self, documents: list[str]) -> list[int]:
        return [1 if probability >= self.threshold else 0 for probability in self.predict_proba(documents)]


def evaluate_predictions(rows: list[dict], predictions: list[int]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    return {
        "overall": summarize(labels, predictions),
        "fpr_clean": negative_fpr(rows, predictions, lambda row: row.get("attack_category") == "none"),
        "fpr_hard_negative": negative_fpr(
            rows, predictions, lambda row: row.get("attack_category") == "hard_negative"
        ),
        "fpr_hard_negative_by_document_source": hard_negative_fpr_by_source(rows, predictions),
        "recall_by_attack_type": grouped_recall(rows, predictions, "attack_type"),
        "recall_by_attack_category": grouped_recall(rows, predictions, "attack_category"),
        "recall_by_source": grouped_recall(rows, predictions, "source"),
        "recall_by_position": grouped_recall(rows, predictions, "position"),
        "recall_by_document_source": grouped_recall(rows, predictions, "document_source"),
    }


def run_lexical_baseline(
    train_rows: list[dict],
    eval_rows: dict[str, list[dict]],
    *,
    max_tokens: int = 512,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model = TruncatedLexicalBaseline(max_tokens=max_tokens, threshold=threshold)
    model.fit(
        [row["document"] for row in train_rows],
        [int(row["label"]) for row in train_rows],
    )
    report: dict[str, Any] = {
        "model": "truncated-lexical",
        "input": "document",
        "uses_user_task": False,
        "max_tokens": max_tokens,
        "threshold": threshold,
        "n_train": len(train_rows),
        "splits": {},
    }
    for name, rows in eval_rows.items():
        if not rows:
            report["splits"][name] = {"overall": None, "rows": 0}
            continue
        predictions = model.predict([row["document"] for row in rows])
        report["splits"][name] = evaluate_predictions(rows, predictions)
        report["splits"][name]["rows"] = len(rows)
    return report


def run_deberta_baseline(
    train_rows: list[dict],
    eval_rows: dict[str, list[dict]],
    *,
    model_name: str = "microsoft/deberta-v3-base",
    max_length: int = 512,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    threshold: float = 0.5,
    seed: int = 2023,
) -> dict[str, Any]:
    """Truncated single-string DeBERTa. The user task is not an input."""
    import os
    import random

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class _Docs(Dataset):
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            row = self.rows[index]
            encoded = tokenizer(
                row["document"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            encoded["labels"] = int(row["label"])
            return {key: torch.tensor(value) for key, value in encoded.items()}

    loader = DataLoader(_Docs(train_rows), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for _epoch in range(epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()

    model.eval()
    report: dict[str, Any] = {
        "model": model_name,
        "input": "document",
        "uses_user_task": False,
        "max_tokens": max_length,
        "threshold": threshold,
        "n_train": len(train_rows),
        "epochs": epochs,
        "splits": {},
    }
    for name, rows in eval_rows.items():
        predictions: list[int] = []
        if rows:
            eval_loader = DataLoader(_Docs(rows), batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch in eval_loader:
                    labels = batch.pop("labels")
                    del labels
                    batch = {key: value.to(device) for key, value in batch.items()}
                    logits = model(**batch).logits
                    probabilities = torch.softmax(logits, dim=-1)[:, 1]
                    predictions.extend(int(probability >= threshold) for probability in probabilities.cpu())
        report["splits"][name] = evaluate_predictions(rows, predictions) if rows else {"overall": None, "rows": 0}
        report["splits"][name]["rows"] = len(rows)
    return report


def _git_head() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _deberta_import_error() -> str | None:
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return str(exc)
    return None


def _deberta_skip_reason(n_train: int) -> str | None:
    missing = _deberta_import_error()
    if missing:
        return missing
    import torch

    if not torch.cuda.is_available() and n_train > 512:
        return (
            "torch is installed but CUDA is not available. "
            f"Skipped the one-epoch DeBERTa run on {n_train} training rows."
        )
    return None


def record_day2_baselines(
    train_rows: list[dict],
    eval_rows: dict[str, list[dict]],
    *,
    max_tokens: int = 512,
    threshold: float = 0.5,
    seed: int = 2023,
    deberta_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the lexical baseline and DeBERTa when its libraries import.

    Scores are written only for runs that finish. A missing DeBERTa install
    leaves that run null and sets status to lexical_only.
    """
    lexical = run_lexical_baseline(
        train_rows,
        eval_rows,
        max_tokens=max_tokens,
        threshold=threshold,
    )
    deberta_report = None
    deberta_note = _deberta_skip_reason(len(train_rows))
    if deberta_note is None:
        deberta = deberta_config or {}
        deberta_report = run_deberta_baseline(
            train_rows,
            eval_rows,
            model_name=str(deberta.get("model_name", "microsoft/deberta-v3-base")),
            max_length=int(deberta.get("max_length", max_tokens)),
            epochs=int(deberta.get("epochs", 1)),
            learning_rate=float(deberta.get("learning_rate", 2e-5)),
            batch_size=int(deberta.get("batch_size", 8)),
            threshold=threshold,
            seed=int(deberta.get("seed", seed)),
        )
    if lexical and deberta_report:
        status = "deberta+lexical"
    elif lexical:
        status = "lexical_only"
    else:
        status = "not_run"
    return {
        "schema_version": "1.1",
        "status": status,
        "threshold": threshold,
        "seed": seed,
        "input": "document",
        "uses_user_task": False,
        "max_tokens": max_tokens,
        "row_counts": {
            "train": len(train_rows),
            "val": len(eval_rows.get("val") or []),
            "test": len(eval_rows.get("test") or []),
        },
        "git_head": _git_head(),
        "deberta_note": deberta_note,
        "runs": {"lexical": lexical, "deberta": deberta_report},
    }


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    return (repo_root / path).resolve()


def run_from_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    data_dir = _resolve(config_path, config.get("data_dir", "data/processed"))
    train_rows = read_jsonl(data_dir / "train.jsonl")
    eval_rows = {
        "val": read_jsonl(data_dir / "val.jsonl") if (data_dir / "val.jsonl").is_file() else [],
        "test": read_jsonl(data_dir / "test.jsonl"),
    }
    model_name = str(config.get("model", "lexical"))
    threshold = float(config.get("threshold", 0.5))
    if config.get("record"):
        recorded = record_day2_baselines(
            train_rows,
            eval_rows,
            max_tokens=int(config.get("max_tokens", 512)),
            threshold=threshold,
            seed=int((config.get("deberta") or {}).get("seed", 2023)),
            deberta_config=config.get("deberta") or {},
        )
        record_path = _resolve(config_path, str(config["record"]))
        record_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(recorded, indent=2)
        record_path.write_text(payload, encoding="utf-8")
        output = _resolve(config_path, config.get("output", "data/processed/baseline_metrics.json"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        test_run = recorded["runs"]["lexical"]["splits"]["test"]["overall"]
        print(
            "recorded baseline status={status} test recall={recall} fpr={fpr} -> {path}".format(
                status=recorded["status"],
                recall=None if test_run is None else test_run["recall"],
                fpr=None if test_run is None else test_run["fpr"],
                path=record_path,
            )
        )
        return recorded
    if model_name == "lexical":
        report = run_lexical_baseline(
            train_rows,
            eval_rows,
            max_tokens=int(config.get("max_tokens", 512)),
            threshold=threshold,
        )
    elif model_name == "deberta":
        deberta = config.get("deberta") or {}
        report = run_deberta_baseline(
            train_rows,
            eval_rows,
            model_name=str(deberta.get("model_name", "microsoft/deberta-v3-base")),
            max_length=int(deberta.get("max_length", 512)),
            epochs=int(deberta.get("epochs", 1)),
            learning_rate=float(deberta.get("learning_rate", 2e-5)),
            batch_size=int(deberta.get("batch_size", 8)),
            threshold=threshold,
            seed=int(deberta.get("seed", 2023)),
        )
    else:
        raise ValueError("baseline model must be 'lexical' or 'deberta'")
    output = _resolve(config_path, config.get("output", "data/processed/baseline_metrics.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    test_summary = report["splits"]["test"]["overall"]
    print(
        "baseline {model} test recall={recall} fpr={fpr} -> {output}".format(
            model=report["model"],
            recall=None if test_summary is None else test_summary["recall"],
            fpr=None if test_summary is None else test_summary["fpr"],
            output=output,
        )
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the truncated single-string baseline.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_from_config(config, args.config.resolve())


if __name__ == "__main__":
    main()
