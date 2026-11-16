"""Obfuscated held-out set: span rewrite, leakage, and eval protocol."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "obfuscated_heldout.yaml"
FAMILIES = ["base64", "hex", "leetspeak", "zero_width", "homoglyph"]


def _payload_at(document: str, spans: list[tuple[int, int]]) -> str:
    return "".join(document[start:end] for start, end in spans)


def test_rewrite_roundtrip_and_context() -> None:
    from sudarshana.data.obfuscation import apply, context_text, rewrite_document

    left = "PAYLOAD1"
    right = "PAYLOAD2"
    document = "AAAA " + left + " BBBB " + right + " CCCC"
    first = (5, 5 + len(left))
    second = (first[1] + len(" BBBB "), first[1] + len(" BBBB ") + len(right))
    spans = [first, second]
    parent_context = context_text(document, spans)

    assert apply("Test", "leetspeak") == "7357"
    assert apply("a", "homoglyph") != "a"

    for family in FAMILIES:
        rewritten, new_spans = rewrite_document(document, spans, family)
        assert len(new_spans) == 2
        assert context_text(rewritten, new_spans) == parent_context
        assert rewritten[new_spans[0][0] : new_spans[0][1]] == apply(left, family)
        assert rewritten[new_spans[1][0] : new_spans[1][1]] == apply(right, family)

    encoded, encoded_spans = rewrite_document("ATTACK\nrest", [(0, 6)], "base64")
    payload = encoded[encoded_spans[0][0] : encoded_spans[0][1]]
    assert base64.b64decode(payload).decode("utf-8") == "ATTACK"
    assert encoded.endswith("\nrest")

    hex_doc, hex_spans = rewrite_document("keep PAYLOAD", [(5, 12)], "hex")
    assert bytes.fromhex(hex_doc[hex_spans[0][0] : hex_spans[0][1]]).decode("utf-8") == "PAYLOAD"
    assert hex_doc.startswith("keep ")

    hidden, hidden_spans = rewrite_document("ab", [(0, 2)], "zero_width")
    assert hidden[hidden_spans[0][0] : hidden_spans[0][1]].replace("\u200b", "") == "ab"

    with pytest.raises(ValueError, match="overlap"):
        rewrite_document("abcdef", [(0, 3), (2, 5)], "hex")


def test_reserved_family_rejected() -> None:
    from sudarshana.data.obfuscation import assert_train_family

    assert assert_train_family("base64") == "base64"
    with pytest.raises(ValueError, match="reserved"):
        assert_train_family("homoglyph")
    with pytest.raises(ValueError, match="unknown"):
        assert_train_family("rot13")


def test_build_rejects_non_test_source(tmp_path: Path) -> None:
    from sudarshana.data.build_obfuscated import build_from_config

    config = {
        "source": "data/splits/train.jsonl",
        "output": str(tmp_path / "obfuscated.jsonl"),
        "families": ["base64"],
        "seed": 1,
    }
    with pytest.raises(ValueError, match="test.jsonl"):
        build_from_config(config, CONFIG_PATH)


def test_full_build_eval_and_determinism(tmp_path: Path) -> None:
    from sudarshana.data.build_obfuscated import build_from_config
    from sudarshana.data.io import read_jsonl
    from sudarshana.data.obfuscated_validate import validate_obfuscated_split
    from sudarshana.data.obfuscation import context_text, span_pairs
    from sudarshana.evaluation.heldout import evaluate_heldout, format_heldout_section

    official = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    first = dict(official)
    first["output"] = str(tmp_path / "a.jsonl")
    first["card"] = str(tmp_path / "a.md")
    second = dict(official)
    second["output"] = str(tmp_path / "b.jsonl")
    second["card"] = str(tmp_path / "b.md")

    summary = build_from_config(first, CONFIG_PATH)
    build_from_config(second, CONFIG_PATH)
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert summary["n_records"] == 706
    assert summary["n_injected_parents"] == 125
    assert summary["n_clean"] == 81

    parents = read_jsonl(REPO_ROOT / "data" / "splits" / "test.jsonl")
    rows = read_jsonl(tmp_path / "a.jsonl")
    assert len(rows) == 706
    train_rows = read_jsonl(REPO_ROOT / "data" / "splits" / "train.jsonl")
    val_rows = read_jsonl(REPO_ROOT / "data" / "splits" / "val.jsonl")
    validate_obfuscated_split(
        rows,
        parents,
        families=FAMILIES,
        train_rows=train_rows,
        val_rows=val_rows,
    )

    clean = [row for row in rows if row["obfuscation_family"] == "none"]
    assert len(clean) == 81
    for row in clean:
        parent = parents[row["parent_index"]]
        assert row["document"] == parent["document"]
        assert row["injection_span"] == parent["injection_span"]
        assert int(row["label"]) == 0

    changed = {family: False for family in FAMILIES}
    for row in rows:
        if int(row["label"]) != 1:
            continue
        parent = parents[row["parent_index"]]
        parent_spans = span_pairs(parent["injection_span"])
        child_spans = span_pairs(row["injection_span"])
        assert context_text(row["document"], child_spans) == context_text(parent["document"], parent_spans)
        family = row["obfuscation_family"]
        original = _payload_at(parent["document"], parent_spans)
        updated = _payload_at(row["document"], child_spans)
        if updated != original:
            changed[family] = True
    assert all(changed.values())

    broken = [dict(row) for row in rows]
    injected = next(row for row in broken if int(row["label"]) == 1)
    injected["document"] = injected["document"] + "x"
    with pytest.raises(ValueError):
        validate_obfuscated_split(
            broken,
            parents,
            families=FAMILIES,
            train_rows=train_rows,
            val_rows=val_rows,
        )

    result = evaluate_heldout(str(tmp_path / "a.jsonl"))
    assert result["status"] == "OK"
    assert result["n_records"] == 706
    assert result["families"] == FAMILIES
    for family in FAMILIES:
        oracle = result["predictors"]["oracle"]["slicing"]["recall_by_obfuscation_family"][family]
        none = result["predictors"]["none"]["slicing"]["recall_by_obfuscation_family"][family]
        assert oracle["recall"] == pytest.approx(1.0)
        assert oracle["support"] == 125
        assert none["recall"] == pytest.approx(0.0)
        paired = result["predictors"]["oracle"]["paired"]["by_family"][family]
        assert paired["child_recall"] == pytest.approx(1.0)
        assert paired["parent_recall"] == pytest.approx(1.0)
        assert paired["recall_delta"] == pytest.approx(0.0)
        assert result["predictors"]["none"]["paired"]["by_family"][family]["recall_delta"] == pytest.approx(0.0)

    skipped = evaluate_heldout(str(tmp_path / "missing.jsonl"))
    assert skipped["status"] == "SKIPPED"
    assert skipped["reason"] == "no obfuscated set yet; P1/P2 deliverable"
    assert "SKIPPED" in format_heldout_section(skipped)
    assert "base64" in format_heldout_section(result)


def test_primary_only_drops_sibling_rows() -> None:
    from sudarshana.data.build_obfuscated import select_parents

    rows = [
        {"id": "parent", "label": 1},
        {"id": "parent::counterfactual", "label": 0},
        {"id": "hn::mismatched_task", "label": 0},
    ]
    assert [row["id"] for row in select_parents(rows, primary_only=True)] == ["parent"]
    assert len(select_parents(rows, primary_only=False)) == 3
