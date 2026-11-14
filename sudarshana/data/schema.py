"""Frozen Day-2 record contract (schema 1.1).

Field names were frozen after hard negatives landed. A new field or a new
source requires a new schema version and a matching edit to ``contract.json``.

Every source is written as one JSON object with the same keys. The five fields
the rest of the pipeline trains and evaluates on are:

    user_task, document, label, injection_span, attack_type

`label` is 1 when `document` contains an injected instruction and 0 when it
does not. `injection_span` is null for clean documents. For injected documents
it is a list of character offsets into `document`, each ``{"start", "end"}``,
and ``document[start:end]`` is injected text.

The extra keys are part of the same contract. Splits are by `attack_type_id`
and `document_source`, never by shuffling rows. `attack_type` is the public
name (a BIPIA attack type, an AgentDojo injection task, or a TensorTrust
family). `attack_type_id` distinguishes collisions such as the two different
BIPIA "Language Translation" templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "1.1"

CORE_FIELDS = (
    "user_task",
    "document",
    "label",
    "injection_span",
    "attack_type",
)

METADATA_FIELDS = (
    "schema_version",
    "id",
    "source",
    "document_source",
    "attack_type_id",
    "attack_category",
    "task_name",
    "modality",
    "position",
    "document_group",
    "split",
    "reference_answer",
)

UNIFIED_FIELDS = CORE_FIELDS + METADATA_FIELDS

SOURCES = ("bipia", "agentdojo", "tensortrust", "hard_negative")
SPLITS = ("train", "val", "test")
LABELS = (0, 1)
CLEAN_ATTACK_CATEGORIES = ("none", "hard_negative")


@dataclass
class UnifiedExample:
    user_task: str
    document: str
    label: int
    injection_span: list[dict[str, int]] | None
    attack_type: str
    id: str
    source: str
    document_source: str
    attack_type_id: str
    attack_category: str
    task_name: str
    modality: str
    position: str | None
    document_group: str
    split: str = ""
    reference_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "user_task": self.user_task,
            "document": self.document,
            "label": self.label,
            "injection_span": self.injection_span,
            "attack_type": self.attack_type,
            "attack_type_id": self.attack_type_id,
            "attack_category": self.attack_category,
            "source": self.source,
            "document_source": self.document_source,
            "task_name": self.task_name,
            "modality": self.modality,
            "position": self.position,
            "document_group": self.document_group,
            "split": self.split,
            "reference_answer": self.reference_answer,
        }


def _span_list(value: Any) -> list[dict[str, int]] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise TypeError("injection_span must be null or a list of {start, end}")
    spans: list[dict[str, int]] = []
    for item in value:
        if not isinstance(item, dict) or "start" not in item or "end" not in item:
            raise TypeError("each injection span needs integer start and end")
        spans.append({"start": int(item["start"]), "end": int(item["end"])})
    return spans


def example_from_dict(row: dict[str, Any]) -> UnifiedExample:
    return UnifiedExample(
        user_task=str(row["user_task"]),
        document=str(row["document"]),
        label=int(row["label"]),
        injection_span=_span_list(row.get("injection_span")),
        attack_type=str(row["attack_type"]),
        id=str(row["id"]),
        source=str(row["source"]),
        document_source=str(row["document_source"]),
        attack_type_id=str(row["attack_type_id"]),
        attack_category=str(row["attack_category"]),
        task_name=str(row["task_name"]),
        modality=str(row["modality"]),
        position=None if row.get("position") is None else str(row["position"]),
        document_group=str(row["document_group"]),
        split=str(row.get("split") or ""),
        reference_answer=None
        if row.get("reference_answer") is None
        else str(row["reference_answer"]),
    )


def validate_example(row: dict[str, Any]) -> None:
    missing = [key for key in UNIFIED_FIELDS if key not in row]
    if missing:
        raise ValueError(f"record {row.get('id')} missing fields: {missing}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"record {row.get('id')} has schema {row['schema_version']}, expected {SCHEMA_VERSION}"
        )
    if row["source"] not in SOURCES:
        raise ValueError(f"record {row.get('id')} has unknown source {row['source']}")
    if row["label"] not in LABELS:
        raise ValueError(f"record {row.get('id')} has label {row['label']}")
    if row["split"] not in SPLITS:
        raise ValueError(f"record {row.get('id')} has split {row['split']!r}")
    if not isinstance(row["user_task"], str) or not isinstance(row["document"], str):
        raise ValueError(f"record {row.get('id')} user_task and document must be strings")

    span = row["injection_span"]
    if row["label"] == 0:
        if span is not None:
            raise ValueError(f"clean record {row.get('id')} must have null injection_span")
        if row["attack_type"] != "none":
            raise ValueError(f"clean record {row.get('id')} must have attack_type 'none'")
        if row["attack_category"] not in CLEAN_ATTACK_CATEGORIES:
            raise ValueError(
                f"clean record {row.get('id')} has attack_category {row['attack_category']!r}"
            )
        if row["source"] == "hard_negative" and row["attack_category"] != "hard_negative":
            raise ValueError(f"hard-negative record {row.get('id')} must use attack_category 'hard_negative'")
        if row["attack_category"] == "hard_negative" and row["source"] != "hard_negative":
            raise ValueError(f"record {row.get('id')} uses hard_negative category without that source")
        return

    if row["source"] == "hard_negative" or row["attack_category"] == "hard_negative":
        raise ValueError(f"injected record {row.get('id')} cannot be a hard negative")

    if not isinstance(span, list) or not span:
        raise ValueError(f"injected record {row.get('id')} needs at least one span")
    document = row["document"]
    previous_end = -1
    for item in span:
        start = int(item["start"])
        end = int(item["end"])
        if not (0 <= start < end <= len(document)):
            raise ValueError(f"span {(start, end)} is outside document {row.get('id')}")
        if start < previous_end:
            raise ValueError(f"spans overlap or are unsorted in {row.get('id')}")
        if not document[start:end].strip():
            raise ValueError(f"empty injection span in {row.get('id')}")
        previous_end = end


SIBLING_SUFFIXES = ("::counterfactual", "::mismatched_task")


def validate_split(rows: list[dict[str, Any]]) -> None:
    """Reject example-level leakage of attack types and BIPIA test documents.

    A counterfactual or mismatched sibling must keep its parent's split and
    document, so a test attack string cannot move into train.
    """
    for row in rows:
        validate_example(row)

    by_id = {row["id"]: row for row in rows}
    for row in rows:
        suffix = next((item for item in SIBLING_SUFFIXES if row["id"].endswith(item)), None)
        if suffix is None:
            continue
        parent = by_id.get(row["id"][: -len(suffix)])
        if (
            parent is None
            or parent["split"] != row["split"]
            or parent["document"] != row["document"]
        ):
            raise ValueError(f"sibling {row['id']} must keep the parent split and document")

    by_attack: dict[str, set[str]] = {}
    bipia_groups: dict[str, set[str]] = {}
    hard_negative_groups: dict[str, set[str]] = {}
    for row in rows:
        if row["label"] == 1:
            by_attack.setdefault(row["attack_type_id"], set()).add(row["split"])
        if row["source"] == "bipia":
            bipia_groups.setdefault(row["document_group"], set()).add(row["split"])
        if row["source"] == "hard_negative":
            hard_negative_groups.setdefault(row["document_group"], set()).add(row["split"])

    for attack_id, splits in by_attack.items():
        if "train" in splits and "test" in splits:
            raise ValueError(f"attack type {attack_id} is in both train and test")
        if "train" in splits and "val" in splits:
            raise ValueError(f"attack type {attack_id} is in both train and val")
        if "val" in splits and "test" in splits:
            raise ValueError(f"attack type {attack_id} is in both val and test")

    for group, splits in bipia_groups.items():
        if "test" in splits and ("train" in splits or "val" in splits):
            raise ValueError(f"BIPIA document group {group} leaks into train or val")

    for group, splits in hard_negative_groups.items():
        if len(splits) > 1:
            raise ValueError(f"hard-negative document group {group} is in {sorted(splits)}")
