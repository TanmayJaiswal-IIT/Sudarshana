"""Build the unified dataset and write train, val, and test JSONL.

    python -m sudarshana.data.build --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from sudarshana.data.counterfactuals import attach_counterfactuals
from sudarshana.data.io import write_jsonl
from sudarshana.data.schema import SCHEMA_VERSION, UNIFIED_FIELDS, validate_split
from sudarshana.data.sources.agentdojo import load_agentdojo
from sudarshana.data.sources.bipia import POSITIONS, load_bipia
from sudarshana.data.sources.hard_negatives import load_hard_negatives
from sudarshana.data.sources.tensortrust import load_tensortrust
from sudarshana.data.splits import assign_splits, split_report


def _repo_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    return (repo_root / path).resolve()


def build_from_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    raw_root = _repo_path(config_path, config.get("raw_dir", "data/raw"))
    output_dir = _repo_path(config_path, config.get("output_dir", "data/processed"))
    download_missing = bool(config.get("download", True))
    seed = int(config.get("seed", 2023))
    max_contexts = config.get("max_contexts")
    if max_contexts is not None:
        max_contexts = int(max_contexts)

    bipia_cfg = config.get("bipia") or {}
    agent_cfg = config.get("agentdojo") or {}
    trust_cfg = config.get("tensortrust") or {}
    hard_cfg = config.get("hard_negatives") or {}
    pair_cfg = config.get("counterfactuals") or {}
    tasks = list(bipia_cfg.get("tasks") or ["email", "table", "code"])
    positions = tuple(bipia_cfg.get("positions") or POSITIONS)

    examples = []
    examples.extend(
        load_bipia(
            raw_root / "bipia",
            tasks=tasks,
            positions=positions,
            seed=seed,
            max_contexts=max_contexts,
            download_missing=download_missing and bool(bipia_cfg.get("download", True)),
        )
    )
    examples.extend(
        load_agentdojo(
            raw_root / "agentdojo",
            mode=str(agent_cfg.get("mode", "auto")),
            benchmark_version=str(agent_cfg.get("benchmark_version", "v1.2.2")),
            jailbreak=str(agent_cfg.get("jailbreak", "ignore_previous")),
        )
    )
    examples.extend(
        load_tensortrust(
            raw_root / "tensortrust",
            download_missing=download_missing and bool(trust_cfg.get("download", True)),
        )
    )
    hard_max = hard_cfg.get("max_per_category", 400)
    hard_examples, hard_provenance = load_hard_negatives(
        raw_root / "hard_negatives",
        download_missing=download_missing and bool(hard_cfg.get("download", True)),
        max_per_category=None if hard_max is None else int(hard_max),
    )
    examples.extend(hard_examples)
    if not examples:
        raise RuntimeError("no examples were loaded")

    assign_splits(
        examples,
        held_out_suite=str(agent_cfg.get("held_out_suite", "travel")),
        val_injection_task_id=str(agent_cfg.get("val_injection_task_id", "injection_task_0")),
        val_user_task_id=str(agent_cfg.get("val_user_task_id", "user_task_0")),
        tensortrust_val_fraction=float(trust_cfg.get("val_fraction", 0.15)),
        held_out_hard_negative_category=str(hard_cfg.get("held_out_category", "shell")),
        hard_negative_val_fraction=float(hard_cfg.get("val_fraction", 0.15)),
    )
    rows = [example.to_dict() for example in examples]
    flip_categories = pair_cfg.get("flip_categories")
    rows, pair_counts = attach_counterfactuals(
        rows,
        enabled=bool(pair_cfg.get("enabled", True)),
        flip_categories=None if flip_categories is None else list(flip_categories),
        mismatched_hard_negatives=bool(pair_cfg.get("mismatched_hard_negatives", True)),
    )
    validate_split(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        write_jsonl(output_dir / f"{name}.jsonl", [row for row in rows if row["split"] == name])
    report = split_report(rows)
    card = {
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        "fields": list(UNIFIED_FIELDS),
        "core_fields": [
            "user_task",
            "document",
            "label",
            "injection_span",
            "attack_type",
        ],
        "split": (
            "Whole attack types and document sources are assigned together. "
            "BIPIA test attack files and test contexts are test. "
            "One train attack type per BIPIA category is val. "
            "AgentDojo suite 'travel' is test. injection_task_0 is val elsewhere. "
            "TensorTrust prompt_extraction is test. A sorted tail of hijack defenses is val. "
            "Hard-negative category 'shell' is test. A sorted tail of the other "
            "hard-negative documents is val. Counterfactual siblings keep the "
            "parent document and split."
        ),
        "seed": seed,
        "counterfactuals": pair_counts,
        "hard_negative_provenance": hard_provenance,
        "report": report,
    }
    (output_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "data_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(
        "wrote {rows} rows to {output} "
        "(train {train}, val {val}, test {test})".format(
            rows=report["rows"],
            output=output_dir,
            train=report["by_split"]["train"]["rows"],
            val=report["by_split"]["val"]["rows"],
            test=report["by_split"]["test"]["rows"],
        )
    )
    return card


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the unified Sudarshana dataset.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-contexts", type=int, default=None)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.skip_download:
        config["download"] = False
    if args.max_contexts is not None:
        config["max_contexts"] = args.max_contexts
    build_from_config(config, args.config.resolve())


if __name__ == "__main__":
    main()
