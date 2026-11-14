"""Document-level counterfactual pairs inside the frozen schema 1.1 fields.

A flipped sibling keeps the parent document and changes the user task to the
injection text, with label 0. Task-relevant and targeted attacks are not
flipped. A mismatched hard negative keeps label 0 and only changes the task.
"""

from __future__ import annotations

from typing import Any

COUNTERFACTUAL_SUFFIX = "::counterfactual"
MISMATCHED_SUFFIX = "::mismatched_task"
TASK_AGREEING = ("task_relevant", "targeted")
DEFAULT_FLIP_CATEGORIES = (
    "task_irrelevant",
    "code",
    "agent_injection",
    "prompt_hijack",
    "prompt_extraction",
)

# Fixed tasks from another category. No text is copied from another document.
MISMATCHED_TASK = {
    "recipe": "Follow the setup instructions.",
    "readme": "What do I need to do?",
    "email": "Show how to use this command.",
    "tutorial": "How do I make this?",
    "shell": "How do I make this?",
}
MISMATCHED_FALLBACK = "Follow the setup instructions."


def span_text(row: dict[str, Any]) -> str:
    document = row["document"]
    parts = [document[int(span["start"]) : int(span["end"])] for span in row["injection_span"]]
    return "\n".join(parts)


def mismatched_user_task(row: dict[str, Any]) -> str:
    category = str(row["document_source"])
    task = MISMATCHED_TASK.get(category, MISMATCHED_FALLBACK)
    if task == row["user_task"] and category in ("tutorial", "shell"):
        task = MISMATCHED_FALLBACK
    if task == row["user_task"]:
        task = "What do I need to do?" if task == MISMATCHED_FALLBACK else MISMATCHED_FALLBACK
    return task


def _counterfactual_row(row: dict[str, Any], user_task: str) -> dict[str, Any]:
    sibling = dict(row)
    sibling["id"] = f"{row['id']}{COUNTERFACTUAL_SUFFIX}"
    sibling["user_task"] = user_task
    sibling["label"] = 0
    sibling["injection_span"] = None
    sibling["attack_type"] = "none"
    sibling["attack_category"] = "none"
    sibling["attack_type_id"] = f"counterfactual:{row['attack_type_id']}"
    sibling["position"] = None
    sibling["reference_answer"] = None
    return sibling


def _mismatched_row(row: dict[str, Any]) -> dict[str, Any]:
    sibling = dict(row)
    sibling["id"] = f"{row['id']}{MISMATCHED_SUFFIX}"
    sibling["user_task"] = mismatched_user_task(row)
    return sibling


def attach_counterfactuals(
    rows: list[dict[str, Any]],
    *,
    enabled: bool = True,
    flip_categories: list[str] | tuple[str, ...] | None = None,
    mismatched_hard_negatives: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Append siblings after the parent rows. Parent order is unchanged."""
    counts = {
        "flipped": 0,
        "skipped_identical_task": 0,
        "task_agreeing_unflipped": 0,
        "mismatched_hard_negatives": 0,
    }
    if not enabled:
        return list(rows), counts

    allowed = set(flip_categories or DEFAULT_FLIP_CATEGORIES)
    allowed.difference_update(TASK_AGREEING)
    siblings: list[dict[str, Any]] = []
    for row in rows:
        if "::" in row["id"]:
            continue
        if row["label"] == 1 and row["attack_category"] in TASK_AGREEING:
            counts["task_agreeing_unflipped"] += 1
        if row["label"] == 1 and row["attack_category"] in allowed:
            user_task = span_text(row)
            if user_task == row["user_task"]:
                counts["skipped_identical_task"] += 1
            else:
                siblings.append(_counterfactual_row(row, user_task))
                counts["flipped"] += 1
        if mismatched_hard_negatives and row["source"] == "hard_negative":
            siblings.append(_mismatched_row(row))
            counts["mismatched_hard_negatives"] += 1
    return list(rows) + siblings, counts
