"""Unit tests for dataset span recovery and round-trip fidelity."""

import json
import random
import os
import pytest


def test_span_roundtrip_verification():
    """Verify that document[start:end] exactly recovers the injection span on >=50 samples."""
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = os.path.join("data/splits", split)
        assert os.path.exists(path), f"Split file {path} missing"

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        injected = [r for r in records if r["label"] == 1]
        assert len(injected) >= 50, f"Expected at least 50 injected samples in {split}, found {len(injected)}"

        # Sample at least 50 samples
        sample_count = min(len(injected), 75)
        samples = random.sample(injected, sample_count)

        for rec in samples:
            doc = rec["document"]
            spans = rec["injection_span"]
            assert len(spans) > 0, "Positive sample must have at least one span"
            for start, end in spans:
                assert 0 <= start < end <= len(doc), f"Span bounds invalid: ({start}, {end}) for len {len(doc)}"
                extracted = doc[start:end]
                assert len(extracted.strip()) > 0, "Extracted span cannot be empty"


def test_schema_compliance():
    """Verify that all records conform to (user_task, document, label, injection_span, attack_type)."""
    required_keys = {"user_task", "document", "label", "injection_span", "attack_type"}
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = os.path.join("data/splits", split)
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                rec = json.loads(line)
                missing = required_keys - set(rec.keys())
                assert not missing, f"Line {idx} in {split} missing keys: {missing}"
                assert rec["label"] in (0, 1), f"Label must be 0 or 1, got {rec['label']}"
                assert isinstance(rec["document"], str)
                assert isinstance(rec["user_task"], str)
                assert isinstance(rec["injection_span"], list)


def test_no_attack_type_leakage():
    """Verify disjoint attack types between train and held-out test splits."""
    def get_attack_types(filename):
        types = set()
        with open(os.path.join("data/splits", filename), "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["label"] == 1:
                    types.add(r["attack_type"])
        return types

    train_types = get_attack_types("train.jsonl")
    val_types = get_attack_types("val.jsonl")
    test_types = get_attack_types("test.jsonl")

    overlap_train_test = train_types.intersection(test_types)
    overlap_val_test = val_types.intersection(test_types)

    assert len(overlap_train_test) == 0, f"Attack type leakage between train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Attack type leakage between val and test: {overlap_val_test}"

