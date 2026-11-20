"""Offline unit tests: no database, no Azure. Run: python -m unittest tests.test_logic

Covers the pure logic (response parsing, applying labels, the batch loop, exit codes).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
os.chdir(tempfile.mkdtemp())  # keep logs/ out of the repo

import pandas as pd

from config.exceptions import PipelineError
from services import Batcher, Parser, PromptBuilder, run_classification
from services.classification_orchestrator import _apply_labels, _product_results
from utils.run_guard import run_guarded

KEY = "category_segment"


class FakeClient:
    """Answers every row except those whose id is in `skip`."""

    def __init__(self, skip=()):
        self.skip, self.prompts = set(skip), []

    def send(self, prompt):
        self.prompts.append(prompt)
        ids = [int(i) for i in re.findall(r'"row_id": (\d+)', prompt)]
        items = [{"row_id": i, KEY: "Cheese"} for i in ids if i not in self.skip]
        return "Result: " + json.dumps(items), {"total_tokens": 5}


def make_batcher(client):
    builder = PromptBuilder(["product_name"], "Cols: {columns} {fallback_label} {response_key}", KEY, "Other")
    return Batcher(client, Parser(KEY), builder)


class ParserTests(unittest.TestCase):
    def test_reads_json_array_inside_text(self):
        out = Parser(KEY).parse_classification_response('ok [{"row_id": 1, "%s": " Cheese "}] done' % KEY)
        self.assertEqual(out, [{"row_id": 1, "label": "Cheese"}])

    def test_skips_bad_entries(self):
        text = json.dumps([5, {"row_id": 1}, {"row_id": "x", KEY: "A"}, {"row_id": 2, KEY: ""}, {"row_id": 3, KEY: "B"}])
        self.assertEqual(Parser(KEY).parse_classification_response(text), [{"row_id": 3, "label": "B"}])

    def test_garbage_and_empty_give_nothing(self):
        for text in ("", "no json here", "[1, 2"):
            self.assertEqual(Parser(KEY).parse_classification_response(text), [])


class ApplyLabelsTests(unittest.TestCase):
    def test_only_empty_rows_are_written(self):
        df = pd.DataFrame({"category": [None, "Old"]})
        parsed = [{"row_id": 0, "label": "New"}, {"row_id": 1, "label": "New"}, {"row_id": 9, "label": "New"}]
        applied, rejected, counts = _apply_labels(df, parsed, "category", 1)
        self.assertEqual((applied, rejected, dict(counts)), (1, 2, {"New": 1}))
        self.assertEqual(df["category"].tolist(), ["New", "Old"])

    def test_product_results_marks_missing_rows(self):
        res = _product_results([0, 1], ["a", "b"], [{"row_id": 0, "label": "X"}])
        self.assertEqual(res, [{"product": "a", "segment": "X"}, {"product": "b", "segment": "(unclassified)"}])


class RunClassificationTests(unittest.TestCase):
    def frame(self, n):
        return pd.DataFrame({"id": range(100, 100 + n), "product_name": [f"p{i}" for i in range(n)]})

    def test_classifies_in_batches_and_leaves_unanswered_rows_empty(self):
        df, client = self.frame(7), FakeClient(skip={3})
        run_classification(make_batcher(client), df, "", ["Cheese"], {}, "category", 3, show_console_start=False)
        self.assertEqual(len(client.prompts), 3)  # 7 rows, batch size 3
        self.assertTrue(pd.isna(df.at[3, "category"]))
        self.assertEqual(df["category"].notna().sum(), 6)

    def test_already_classified_rows_are_not_sent(self):
        df = self.frame(4)
        df["category"] = ["Done", None, "Done", None]
        client = FakeClient()
        run_classification(make_batcher(client), df, "", ["Cheese"], {}, "category", 10, show_console_start=False)
        self.assertEqual(len(client.prompts), 1)
        self.assertEqual(df["category"].tolist(), ["Done", "Cheese", "Done", "Cheese"])

    def test_partial_json_written_after_batches(self):
        out = Path(tempfile.mkdtemp()) / "sub" / "partial.json"
        run_classification(make_batcher(FakeClient()), self.frame(2), "", ["Cheese"], {}, "category", 1,
                           partial_output_json=str(out), show_console_start=False)
        self.assertEqual(len(json.loads(out.read_text())), 2)


class RunGuardTests(unittest.TestCase):
    def test_exit_codes(self):
        def boom(exc):
            def run():
                raise exc
            return run
        self.assertEqual(run_guarded(lambda: 0), 0)
        self.assertEqual(run_guarded(boom(KeyboardInterrupt())), 130)
        self.assertEqual(run_guarded(boom(PipelineError("x"))), 1)
        self.assertEqual(run_guarded(boom(RuntimeError("x"))), 1)


if __name__ == "__main__":
    unittest.main()
