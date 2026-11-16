"""Checks for the obfuscated held-out split.

Injected children must reproduce the transform, keep the parent context, and
stay out of train and val. Clean rows are verbatim copies of test rows. A
clean test document, or a whole clean row, that also appears in another split
is a property of the existing split, so that overlap is counted and not rejected.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from sudarshana.data.obfuscation import (
    CLEAN_FAMILY,
    canonical_sha256,
    context_text,
    rewrite_document,
    span_pairs,
)


def document_key(document: str) -> str:
    """Stable identity for a document string. Used so leakage checks need not keep the text."""
    return hashlib.sha256(str(document).encode("utf-8")).hexdigest()


def leakage_sets(rows: Iterable[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Document keys and canonical row hashes for one split."""
    documents: set[str] = set()
    hashes: set[str] = set()
    for row in rows:
        documents.add(document_key(row["document"]))
        hashes.add(canonical_sha256(row))
    return documents, hashes


class ObfuscatedChecker:
    """Accept derived rows one at a time and raise when a row drifts or leaks."""

    def __init__(
        self,
        parents: list[dict[str, Any]],
        families: list[str],
        train_docs: set[str],
        val_docs: set[str],
        train_hashes: set[str],
        val_hashes: set[str],
    ) -> None:
        if not families:
            raise ValueError("obfuscated held-out needs at least one family")
        self.parents = parents
        self.families = families
        self.train_docs = train_docs
        self.val_docs = val_docs
        self.train_hashes = train_hashes
        self.val_hashes = val_hashes
        self.parent_hashes = [canonical_sha256(row) for row in parents]
        if len(set(self.parent_hashes)) != len(self.parent_hashes):
            raise ValueError("test parents do not have unique canonical hashes")
        self.by_hash = dict(zip(self.parent_hashes, parents))
        self.expected_ids: set[str] = set()
        for index, parent in enumerate(parents):
            label = int(parent["label"])
            if label == 0:
                self.expected_ids.add(f"{index}::obf::{CLEAN_FAMILY}")
            elif label == 1:
                if not span_pairs(parent.get("injection_span")):
                    raise ValueError(f"injected parent {index} has no span")
                for family in families:
                    self.expected_ids.add(f"{index}::obf::{family}")
            else:
                raise ValueError(f"parent {index} has label {parent['label']!r}")
        self.seen_ids: set[str] = set()
        self.counts: dict[str, int] = {CLEAN_FAMILY: 0}
        for family in families:
            self.counts[family] = 0
        self.unchanged: dict[str, int] = {family: 0 for family in families}
        self.clean_overlap = {"train": 0, "val": 0}
        self.n_records = 0

    def accept(self, row: dict[str, Any]) -> None:
        row_id = str(row.get("id"))
        if row_id in self.seen_ids:
            raise ValueError(f"duplicate obfuscated id {row_id}")
        self.seen_ids.add(row_id)
        self.n_records += 1

        family = row.get("obfuscation_family")
        parent_index = row.get("parent_index")
        digest = row.get("parent_sha256")
        if not isinstance(parent_index, int) or not (0 <= parent_index < len(self.parents)):
            raise ValueError(f"{row_id} has parent_index {parent_index!r}")
        parent = self.parents[parent_index]
        if digest != self.parent_hashes[parent_index] or digest not in self.by_hash:
            raise ValueError(f"{row_id} parent_sha256 does not match test row {parent_index}")
        for key in ("user_task", "label", "attack_type", "position"):
            if key in parent and row.get(key) != parent.get(key):
                raise ValueError(f"{row_id} changed parent field {key}")

        label = int(parent["label"])
        if label == 0:
            if family != CLEAN_FAMILY or row_id != f"{parent_index}::obf::{CLEAN_FAMILY}":
                raise ValueError(f"clean parent {parent_index} must be copied once as {CLEAN_FAMILY}")
            if row["document"] != parent["document"] or row.get("injection_span") != parent.get("injection_span"):
                raise ValueError(f"{row_id} clean control is not a verbatim copy")
            self.counts[CLEAN_FAMILY] += 1
            doc_key = document_key(row["document"])
            if doc_key in self.train_docs:
                self.clean_overlap["train"] += 1
            if doc_key in self.val_docs:
                self.clean_overlap["val"] += 1
            return

        if family not in self.families or row_id != f"{parent_index}::obf::{family}":
            raise ValueError(f"{row_id} is not an injected child of the requested families")
        pairs = span_pairs(parent.get("injection_span"))
        rewritten, new_pairs = rewrite_document(parent["document"], pairs, str(family))
        if row["document"] != rewritten or span_pairs(row.get("injection_span")) != new_pairs:
            raise ValueError(f"{row_id} document or spans do not match family {family}")
        if context_text(row["document"], new_pairs) != context_text(parent["document"], pairs):
            raise ValueError(f"{row_id} changed text outside the payload")
        if digest in self.train_hashes or digest in self.val_hashes:
            raise ValueError(f"{row_id} injected parent row is also in train or val")
        child_key = document_key(row["document"])
        if child_key in self.train_docs or child_key in self.val_docs:
            raise ValueError(f"{row_id} injected document appears in train or val")
        self.counts[str(family)] += 1
        original_payloads = [parent["document"][start:end] for start, end in pairs]
        new_payloads = [row["document"][start:end] for start, end in new_pairs]
        if original_payloads == new_payloads:
            self.unchanged[str(family)] += 1

    def finish(self) -> dict[str, Any]:
        if self.seen_ids != self.expected_ids:
            missing = sorted(self.expected_ids - self.seen_ids)[:5]
            extra = sorted(self.seen_ids - self.expected_ids)[:5]
            raise ValueError(f"obfuscated ids do not match the test parents (missing {missing}, extra {extra})")
        injected_parents = sum(1 for row in self.parents if int(row["label"]) == 1)
        clean_parents = sum(1 for row in self.parents if int(row["label"]) == 0)
        return {
            "n_records": self.n_records,
            "n_injected_parents": injected_parents,
            "n_clean": clean_parents,
            "counts": self.counts,
            "unchanged_payloads": self.unchanged,
            "clean_document_overlap": self.clean_overlap,
        }


def validate_obfuscated_split(
    rows: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    *,
    families: list[str],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Raise ``ValueError`` when the derived split drifts or leaks.

    Returns counts the card and the builder print.
    """
    train_docs, train_hashes = leakage_sets(train_rows)
    val_docs, val_hashes = leakage_sets(val_rows)
    checker = ObfuscatedChecker(
        parents,
        families,
        train_docs,
        val_docs,
        train_hashes,
        val_hashes,
    )
    for row in rows:
        checker.accept(row)
    return checker.finish()
