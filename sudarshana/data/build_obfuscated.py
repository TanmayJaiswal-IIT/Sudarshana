"""Build the obfuscated held-out split from the frozen test split.

    python -m sudarshana.data.build_obfuscated --config configs/obfuscated_heldout.yaml

The evaluator does not call this module. Training must not read the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterator

import yaml

from sudarshana.data.io import iter_jsonl
from sudarshana.data.obfuscated_validate import ObfuscatedChecker, leakage_sets
from sudarshana.data.obfuscation import (
    CLEAN_FAMILY,
    EVAL_FAMILIES,
    RESERVED_FAMILIES,
    canonical_sha256,
    rewrite_document,
    span_pairs,
    span_style,
    spans_as,
)


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


def _git_head(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _parse_families(raw: Any) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("config families must be a non-empty list")
    families: list[str] = []
    for item in raw:
        family = str(item)
        if family == CLEAN_FAMILY or family not in EVAL_FAMILIES:
            raise ValueError(f"unknown obfuscation family {family!r}")
        if family in families:
            raise ValueError(f"duplicate obfuscation family {family!r}")
        families.append(family)
    return families


def _child(parent: dict[str, Any], index: int, digest: str, family: str) -> dict[str, Any]:
    child = {
        "id": f"{index}::obf::{family}",
        "parent_index": index,
        "parent_sha256": digest,
        "obfuscation_family": family,
    }
    for key, value in parent.items():
        if key not in child:
            child[key] = value
    return child


def select_parents(rows: list[dict[str, Any]], *, primary_only: bool) -> list[dict[str, Any]]:
    """Drop counterfactual and mismatched siblings when ``primary_only`` is set.

    Those ids contain ``::``. Their documents still hold the attack text while
    the label is 0, so they are not clean controls and not extra attacks.
    """
    if not primary_only:
        return list(rows)
    return [row for row in rows if "::" not in str(row.get("id") or "")]


def iter_derived(parents: list[dict[str, Any]], families: list[str]) -> Iterator[dict[str, Any]]:
    """One clean copy per label-0 parent, and one child per family for label 1."""
    for index, parent in enumerate(parents):
        digest = canonical_sha256(parent)
        label = int(parent["label"])
        if label == 0:
            yield _child(parent, index, digest, CLEAN_FAMILY)
            continue
        if label != 1:
            raise ValueError(f"parent {index} has label {parent['label']!r}")
        pairs = span_pairs(parent.get("injection_span"))
        if not pairs:
            raise ValueError(f"injected parent {index} has no span")
        style = span_style(parent.get("injection_span"))
        for family in families:
            rewritten, new_pairs = rewrite_document(str(parent["document"]), pairs, family)
            child = _child(parent, index, digest, family)
            child["document"] = rewritten
            child["injection_span"] = spans_as(new_pairs, style)
            yield child


def derive_rows(parents: list[dict[str, Any]], families: list[str]) -> list[dict[str, Any]]:
    """Materialise :func:`iter_derived`."""
    return list(iter_derived(parents, families))


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _card_text(
    *,
    source: Path,
    source_sha256: str,
    seed: int,
    git_revision: str,
    families: list[str],
    summary: dict[str, Any],
) -> str:
    count_rows = ["| Family | Rows | Unchanged payloads |", "|--------|------|--------------------|"]
    counts = summary["counts"]
    unchanged = summary["unchanged_payloads"]
    count_rows.append(f"| {CLEAN_FAMILY} | {counts[CLEAN_FAMILY]} | — |")
    for family in families:
        count_rows.append(f"| {family} | {counts[family]} | {unchanged[family]} |")
    overlap = summary["clean_document_overlap"]
    reserved = ", ".join(f"`{family}`" for family in RESERVED_FAMILIES)
    return "\n".join(
        [
            "# Obfuscated held-out card",
            "",
            "Surface-obfuscated copy of the test split. Evaluation reads this file.",
            "Training must not. Reserved families must not be used as training augmentation.",
            "",
            f"- Source: `{source.as_posix()}`",
            f"- Source sha256: `{source_sha256}`",
            f"- Seed: {seed} (recorded; the current transforms do not sample)",
            f"- Git revision: `{git_revision}`",
            f"- Rows: {summary['n_records']}",
            f"- Injected parents: {summary['n_injected_parents']}",
            f"- Clean controls: {summary['n_clean']}",
            f"- Reserved families: {reserved}",
            f"- Clean controls whose document text also appears in train: {overlap['train']}",
            f"- Clean controls whose document text also appears in val: {overlap['val']}",
            f"- Primary rows only: {summary.get('primary_only', False)}",
            "",
            "## Counts by family",
            "",
            *count_rows,
            "",
            "Protocol: `docs/eval_protocol_obfuscated.md`",
            "",
        ]
    )


def build_from_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Write the JSONL and the card. Returns the validator summary plus paths."""
    source = _resolve(config_path, str(config["source"]))
    output = _resolve(config_path, str(config["output"]))
    card_value = config.get("card") or str(output.with_name("OBFUSCATED_CARD.md"))
    card_path = _resolve(config_path, str(card_value))
    if source.name != "test.jsonl":
        raise ValueError("obfuscated held-out source must be test.jsonl")

    families = _parse_families(config.get("families"))
    seed = int(config.get("seed", 20261115))
    primary_only = bool(config.get("primary_only", False))
    train_path = source.parent / "train.jsonl"
    val_path = source.parent / "val.jsonl"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("train.jsonl and val.jsonl must sit beside test.jsonl for the leakage check")
    print(f"reading parents from {source}", flush=True)
    parents = select_parents(iter_jsonl(source), primary_only=primary_only)
    print(f"parents {len(parents)}; indexing train for leakage", flush=True)
    train_docs, train_hashes = leakage_sets(iter_jsonl(train_path))
    print(f"indexing val for leakage ({len(train_docs)} distinct train documents)", flush=True)
    val_docs, val_hashes = leakage_sets(iter_jsonl(val_path))
    print("writing obfuscated rows", flush=True)
    checker = ObfuscatedChecker(parents, families, train_docs, val_docs, train_hashes, val_hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            written = 0
            for child in iter_derived(parents, families):
                checker.accept(child)
                handle.write(json.dumps(child, ensure_ascii=False) + "\n")
                written += 1
                if written % 20000 == 0:
                    print(f"validated {written} obfuscated rows", flush=True)
        summary = checker.finish()
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(output)
    summary["primary_only"] = primary_only
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    card_path.write_text(
        _card_text(
            source=Path(_display_path(source, _repo_root(config_path))),
            source_sha256=source_sha,
            seed=seed,
            git_revision=_git_head(_repo_root(config_path)),
            families=families,
            summary=summary,
        ),
        encoding="utf-8",
        newline="\n",
    )
    summary = dict(summary)
    summary["output"] = str(output)
    summary["card"] = str(card_path)
    summary["seed"] = seed
    summary["families"] = list(families)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the obfuscated held-out split from test.jsonl.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = build_from_config(config, config_path)
    print(
        "wrote {n} rows to {output} (injected parents {injected}, clean {clean})".format(
            n=summary["n_records"],
            output=summary["output"],
            injected=summary["n_injected_parents"],
            clean=summary["n_clean"],
        )
    )


if __name__ == "__main__":
    main()
