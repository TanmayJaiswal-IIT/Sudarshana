"""Counterfactual pairs and FPR at a fixed recall."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sudarshana.data.build import build_from_config
from sudarshana.data.counterfactuals import attach_counterfactuals
from sudarshana.data.schema import SCHEMA_VERSION, validate_split
from sudarshana.evaluation.baseline import TruncatedLexicalBaseline
from sudarshana.evaluation.fixed_recall import pair_text, run_fixed_recall
from sudarshana.evaluation.metrics import predictions_at_threshold, summarize, threshold_at_recall

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw"
CONTRACT = Path(__file__).resolve().parents[1] / "sudarshana" / "data" / "contract.json"


def _config(output_dir: Path) -> dict:
    return {
        "seed": 2023,
        "download": False,
        "raw_dir": str(FIXTURES),
        "output_dir": str(output_dir),
        "bipia": {"download": False, "tasks": ["email", "code"], "positions": ["end", "start", "middle"]},
        "agentdojo": {"mode": "jsonl", "held_out_suite": "travel"},
        "tensortrust": {"download": False, "val_fraction": 0.15},
        "hard_negatives": {"download": False},
        "counterfactuals": {"enabled": True, "mismatched_hard_negatives": True},
    }


def _positive(category: str, user_task: str = "Answer the question.") -> dict:
    document = "Context before. PAYLOAD after."
    start = document.index("PAYLOAD")
    return {
        "schema_version": "1.1",
        "id": f"row-{category}",
        "user_task": user_task,
        "document": document,
        "label": 1,
        "injection_span": [{"start": start, "end": start + len("PAYLOAD")}],
        "attack_type": category,
        "attack_type_id": f"attack-{category}",
        "attack_category": category,
        "source": "bipia",
        "document_source": "email",
        "task_name": "email",
        "modality": "text",
        "position": "end",
        "document_group": f"group-{category}",
        "split": "test",
        "reference_answer": "answer",
    }


class CounterfactualTests(unittest.TestCase):
    def test_task_irrelevant_flips_and_agreeing_categories_do_not(self) -> None:
        irrelevant = _positive("task_irrelevant")
        relevant = _positive("task_relevant")
        targeted = _positive("targeted")
        rows, counts = attach_counterfactuals([irrelevant, relevant, targeted])
        self.assertEqual(counts["flipped"], 1)
        self.assertEqual(counts["task_agreeing_unflipped"], 2)
        self.assertEqual(len(rows), 4)
        sibling = rows[-1]
        self.assertEqual(sibling["id"], "row-task_irrelevant::counterfactual")
        self.assertEqual(sibling["document"], irrelevant["document"])
        self.assertEqual(sibling["user_task"], "PAYLOAD")
        self.assertEqual(sibling["label"], 0)
        self.assertIsNone(sibling["injection_span"])
        self.assertEqual(sibling["attack_type"], "none")
        self.assertEqual(sibling["split"], "test")
        self.assertNotIn("row-task_relevant::counterfactual", {row["id"] for row in rows})
        self.assertNotIn("row-targeted::counterfactual", {row["id"] for row in rows})

    def test_identical_task_is_skipped(self) -> None:
        row = _positive("task_irrelevant", user_task="PAYLOAD")
        rows, counts = attach_counterfactuals([row])
        self.assertEqual(counts["skipped_identical_task"], 1)
        self.assertEqual(counts["flipped"], 0)
        self.assertEqual(len(rows), 1)

    def test_fixture_siblings_stay_with_the_parent_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            card = build_from_config(_config(Path(tmp)), Path("unused.yaml"))
            rows = []
            for name in ("train", "val", "test"):
                path = Path(tmp) / f"{name}.jsonl"
                rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
            validate_split(rows)
            self.assertEqual(card["schema_version"], "1.1")
            self.assertTrue(card["frozen"])
            self.assertGreater(card["counterfactuals"]["flipped"], 0)
            self.assertGreater(card["counterfactuals"]["mismatched_hard_negatives"], 0)
            by_id = {row["id"]: row for row in rows}

            parent = next(
                row for row in rows if row["label"] == 1 and row["attack_type"] == "Task Automation"
            )
            sibling = by_id[parent["id"] + "::counterfactual"]
            self.assertEqual(parent["split"], "test")
            self.assertEqual(sibling["split"], "test")
            self.assertEqual(sibling["document"], parent["document"])
            self.assertEqual(sibling["label"], 0)
            self.assertIsNone(sibling["injection_span"])
            train_ids = {row["id"] for row in rows if row["split"] == "train"}
            self.assertNotIn(sibling["id"], train_ids)

            shell = next(
                row
                for row in rows
                if row["source"] == "hard_negative"
                and row["document_source"] == "shell"
                and "::" not in row["id"]
            )
            mismatched = by_id[shell["id"] + "::mismatched_task"]
            self.assertEqual(shell["split"], "test")
            self.assertEqual(mismatched["split"], "test")
            self.assertEqual(mismatched["label"], 0)
            self.assertEqual(mismatched["document"], shell["document"])
            self.assertNotEqual(mismatched["user_task"], shell["user_task"])

            drifting = json.loads(json.dumps(rows))
            moved = next(row for row in drifting if row["id"].endswith("::counterfactual") and row["split"] == "train")
            moved["split"] = "val"
            with self.assertRaises(ValueError):
                validate_split(drifting)


class FixedRecallTests(unittest.TestCase):
    def test_threshold_rule_and_empty_positives(self) -> None:
        probabilities = [0.1, 0.4, 0.4, 0.9]
        labels = [1, 1, 1, 1]
        self.assertEqual(threshold_at_recall(probabilities, labels, 0.95), 0.1)
        tied = threshold_at_recall(probabilities, labels, 0.5)
        self.assertEqual(tied, 0.4)
        caught = predictions_at_threshold(probabilities, tied)
        self.assertEqual(summarize(labels, caught)["recall"], 0.75)
        self.assertIsNone(threshold_at_recall([0.2, 0.8], [0, 0], 0.95))

    def test_pair_scores_use_the_task_and_document_scores_do_not(self) -> None:
        shared = "shared document text"
        rows = [
            {"user_task": "ALPHAWORD please", "document": shared, "label": 1},
            {"user_task": "BETAWORD please", "document": shared, "label": 0},
        ]
        document_model = TruncatedLexicalBaseline().fit(
            [row["document"] for row in rows],
            [row["label"] for row in rows],
        )
        self.assertEqual(
            document_model.score(rows[0]["document"]),
            document_model.score(rows[1]["document"]),
        )
        pair_model = TruncatedLexicalBaseline().fit(
            [pair_text(row) for row in rows],
            [row["label"] for row in rows],
        )
        self.assertNotEqual(pair_model.score(pair_text(rows[0])), pair_model.score(pair_text(rows[1])))

    def test_fixture_report_and_frozen_contract(self) -> None:
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, "1.1")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_from_config(_config(root / "processed"), Path("unused.yaml"))
            splits = {
                name: [
                    json.loads(line)
                    for line in (root / "processed" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                for name in ("train", "val", "test")
            }
            report = run_fixed_recall(splits["train"], {"val": splits["val"], "test": splits["test"]})
        self.assertEqual(report["status"], "lexical_pair+lexical_document")
        self.assertEqual(report["target_recall"], 0.95)
        pair_test = report["runs"]["pair"]["splits"]["test"]
        self.assertGreaterEqual(report["runs"]["pair"]["achieved_val_recall"], 0.95)
        self.assertIn("fpr_hard_negative", pair_test)
        self.assertIn("fpr_counterfactual", pair_test)
        self.assertIn("fpr_mismatched_hard_negative", pair_test)
        self.assertGreater(pair_test["fpr_hard_negative"]["support"], 0)
        self.assertTrue(report["runs"]["pair"]["uses_user_task"])
        self.assertFalse(report["runs"]["document"]["uses_user_task"])


if __name__ == "__main__":
    unittest.main()
