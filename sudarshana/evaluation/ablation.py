"""Coordinate the six-rung ablation ladder at one false-positive cap.

A rung runs only when its component is present and every earlier rung
returned OK. Missing components stay NOT_RUN. The document and pair lexical
models are also calibrated on their own in ``available_arms``. That table is
not an incremental gain.

    python -m sudarshana.evaluation.ablation --config configs/ablation_ladder.yaml
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import yaml

from sudarshana.evaluation.baseline import _git_head
from sudarshana.evaluation.calibrate import (
    REDACTION_THRESHOLD_NOTE,
    build_calibration_from_paths,
    write_json,
    _repo_root,
    _resolve,
)

RUNGS: tuple[dict[str, Any], ...] = (
    {
        "id": "1_baseline",
        "name": "Baseline — truncated, single string",
        "arm": "document",
    },
    {
        "id": "2_normalizer",
        "name": "+ Normalizer",
        "module": "sudarshana.preprocessing.normalizer",
        "attr": "normalize",
        "missing": "no normalizer module",
    },
    {
        "id": "3_chunking",
        "name": "+ Chunking with max pooling",
        "module": "sudarshana.preprocessing.chunker",
        "attr": "chunk_document",
        "missing": "no chunker module",
    },
    {
        "id": "4_pair",
        "name": "+ Pair input",
        "module": "sudarshana.preprocessing.pair_chunks",
        "attr": "score_pair_chunks",
        "missing": (
            "incremental pair rung waits until the normalizer and chunker run; "
            "the lexical pair arm is in available_arms"
        ),
    },
    {
        "id": "5_token_head",
        "name": "+ Token head and redaction",
        "missing": "no token-head checkpoint weights",
    },
    {
        "id": "6_augmentation",
        "name": "+ Obfuscation augmentation",
        "module": "sudarshana.training.augment",
        "attr": "augment_train",
        "missing": "no training augmenter",
    },
)


def _component_ready(rung: dict[str, Any], repo_root: Path) -> bool:
    if rung["id"] == "1_baseline":
        return True
    if rung["id"] == "5_token_head":
        return _token_weights_present(repo_root)
    module_name = rung.get("module")
    attr = rung.get("attr")
    if not module_name or not attr:
        return False
    try:
        module = importlib.import_module(str(module_name))
    except ImportError:
        return False
    return callable(getattr(module, str(attr), None))


def _token_weights_present(repo_root: Path) -> bool:
    root = repo_root / "checkpoints"
    if not root.is_dir():
        return False
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".bin", ".safetensors", ".pt"}:
            return True
    return False


def _rung_record(rung: dict[str, Any], arm: dict[str, Any] | None) -> dict[str, Any]:
    if arm is None:
        return {
            "id": rung["id"],
            "name": rung["name"],
            "status": "NOT_RUN",
            "reason": rung["missing"],
        }
    return {
        "id": rung["id"],
        "name": rung["name"],
        "status": "OK",
        "reason": None,
        "threshold": arm["threshold"],
        "tight": arm["tight"],
        "val": arm["val"],
        "test": arm["test"],
        "hard_negatives": arm["hard_negatives"],
        "obfuscated": arm["obfuscated"],
        "recall_reference": arm["recall_reference"],
    }


def build_ladder(calibration: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Turn a calibration report into the incremental ladder plus the available-arms table."""
    arms = calibration["arms"]
    rungs: list[dict[str, Any]] = []
    previous_ok = True
    for rung in RUNGS:
        ready = _component_ready(rung, repo_root)
        if rung["id"] == "1_baseline" and ready and previous_ok:
            record = _rung_record(rung, arms["document"])
        else:
            blocked = dict(rung)
            if not ready:
                blocked["missing"] = str(rung["missing"])
            elif not previous_ok:
                blocked["missing"] = "earlier rung did not run"
            else:
                blocked["missing"] = "component is present but the stacked scorer is not registered"
            record = _rung_record(blocked, None)
        if record["status"] != "OK":
            previous_ok = False
        rungs.append(record)

    frozen = next((rung["id"] for rung in reversed(rungs) if rung["status"] == "OK"), None)
    return {
        "schema_version": "1.1",
        "target_fpr": calibration["target_fpr"],
        "calibration_split": calibration["calibration_split"],
        "seed": calibration["seed"],
        "max_tokens": calibration["max_tokens"],
        "frozen_rung": frozen,
        "redaction_threshold": calibration["redaction_threshold"],
        "redaction_threshold_note": REDACTION_THRESHOLD_NOTE,
        "git_head": _git_head(),
        "rungs": rungs,
        "available_arms": arms,
    }


def format_ablation_section(ladder: dict[str, Any] | None) -> str:
    """Markdown table for the P5 report."""
    if not ladder:
        return (
            "> Ablation ladder has not been run. "
            "`python -m sudarshana.evaluation.ablation --config configs/ablation_ladder.yaml`"
        )
    lines = [
        (
            f"Target false-positive rate {ladder.get('target_fpr')} on `{ladder.get('calibration_split')}`. "
            f"Frozen rung: `{ladder.get('frozen_rung')}`."
        ),
        (
            f"Redaction cutoff remains {ladder.get('redaction_threshold')} and is not this detection threshold."
        ),
        "",
        "| Rung | Status | Threshold | Val FPR | Val recall | Test recall | Test FPR |",
        "|------|--------|-----------|---------|------------|-------------|----------|",
    ]
    for rung in ladder.get("rungs") or []:
        if rung.get("status") != "OK":
            lines.append(
                f"| {rung.get('name')} | NOT_RUN | — | — | — | — | — |"
            )
            continue
        val = rung.get("val") or {}
        test = rung.get("test") or {}
        lines.append(
            f"| {rung.get('name')} | OK | {rung.get('threshold')} | {val.get('fpr')} | "
            f"{val.get('recall')} | {test.get('recall')} | {test.get('fpr')} |"
        )
    lines.extend(["", "Reasons for rungs that did not run:", ""])
    for rung in ladder.get("rungs") or []:
        if rung.get("status") != "OK":
            lines.append(f"- `{rung.get('id')}`: {rung.get('reason')}")
    lines.extend(
        [
            "",
            "Available arms use the same cap on their own. They are not incremental ladder gains.",
            "",
            "| Arm | Threshold | Val FPR | Val recall | Test recall | Test FPR | Hard-neg FPR |",
            "|-----|-----------|---------|------------|-------------|----------|--------------|",
        ]
    )
    for name, arm in (ladder.get("available_arms") or {}).items():
        val = arm.get("val") or {}
        test = arm.get("test") or {}
        hard = arm.get("hard_negatives") or {}
        lines.append(
            f"| {name} | {arm.get('threshold')} | {val.get('fpr')} | {val.get('recall')} | "
            f"{test.get('recall')} | {test.get('fpr')} | {hard.get('fpr')} |"
        )
    lines.extend(["", "| Arm | Family | N | Parent recall | Child recall | Delta |", "|-----|--------|---|---------------|--------------|-------|"])
    for name, arm in (ladder.get("available_arms") or {}).items():
        families = (arm.get("obfuscated") or {}).get("by_family") or {}
        if not families:
            lines.append(f"| {name} | — | — | — | — | — |")
            continue
        for family, block in families.items():
            lines.append(
                f"| {name} | {family} | {block.get('n')} | {block.get('parent_recall')} | "
                f"{block.get('child_recall')} | {block.get('recall_delta')} |"
            )
    return "\n".join(lines)


def run_from_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    splits_dir = _resolve(config_path, str(config.get("splits_dir", "data/splits")))
    obfuscated = _resolve(config_path, str(config["obfuscated"])) if config.get("obfuscated") else None
    repo_root = _repo_root(config_path)
    calibration = build_calibration_from_paths(
        splits_dir,
        obfuscated,
        target_fpr=float(config.get("target_fpr", 0.05)),
        max_tokens=int(config.get("max_tokens", 512)),
        seed=int(config.get("seed", 2023)),
        target_recall=float(config.get("target_recall_reference", 0.95)),
        repo_root=repo_root,
    )
    if config.get("calibration_output"):
        write_json(_resolve(config_path, str(config["calibration_output"])), calibration)
    ladder = build_ladder(calibration, repo_root)
    output = _resolve(config_path, str(config.get("output", "results/ablation_ladder.json")))
    write_json(output, ladder)
    print(
        "ablation frozen_rung={frozen} target_fpr={target} -> {path}".format(
            frozen=ladder["frozen_rung"],
            target=ladder["target_fpr"],
            path=output,
        )
    )
    return ladder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate at a target FPR and record the ablation ladder.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_from_config(config, config_path)


if __name__ == "__main__":
    main()
