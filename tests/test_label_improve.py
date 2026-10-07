import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logbench import crossllm
from logbench.crossexternal import load_srbh
from logbench.labelimprove import (
    DIRECTIONS, build_payloads, encoding_view, load_prepared, metric_counts,
    nearby_session_payload, text_id, write_outputs, system_prompt_for, SESSION_GUARD,
)


class ImprovedLabellingTests(unittest.TestCase):
    def test_all_directions_are_exposed(self):
        self.assertEqual(set(DIRECTIONS), {"b_to_a", "srbh_to_b", "srbh_to_a"})
        self.assertTrue(all(config["target"] in {"system_a", "system_b"}
                            for config in DIRECTIONS.values()))
        self.assertEqual(DIRECTIONS["srbh_to_a"],
                         {"source": "srbh", "target": "system_a", "method": "nearby_session"})
        self.assertEqual(system_prompt_for("srbh_to_a"), crossllm.SYSTEM + SESSION_GUARD)

    def test_direction_prompts_and_cache_are_isolated(self):
        class Provider:
            def describe(self):
                return {"model": "test"}
        self.assertEqual(system_prompt_for("b_to_a"), crossllm.SYSTEM + SESSION_GUARD)
        self.assertEqual(system_prompt_for("srbh_to_b"), crossllm.SYSTEM)
        self.assertEqual(DIRECTIONS["b_to_a"]["method"], "nearby_session")
        base = crossllm.Annotator(Provider(), [], None,
                                  system_prompt=system_prompt_for("srbh_to_b"))
        changed = crossllm.Annotator(Provider(), [], None,
                                     system_prompt=system_prompt_for("b_to_a"))
        self.assertNotEqual(base.settings_hash, changed.settings_hash)

    def test_encoding_view_keeps_depth_and_tail(self):
        text = "GET /news/%" + "25" * 300 + "2e?x=1 HTTP/1.1"
        view, facts = encoding_view(text)
        self.assertIn("25 repeated 300 times", view)
        self.assertTrue(view.endswith("HTTP/1.1"))
        self.assertIn("300", facts)
        self.assertLessEqual(len(view), 400)

    def test_nearby_context_uses_same_group_and_closest_rows(self):
        rows = [dict(text=f"GET /p{i} HTTP/1.1", group_id="a",
                     timestamp=pd.Timestamp(f"2026-01-01 00:00:{i:02d}"), sample_id=str(i), system="s")
                for i in range(20)]
        frame = pd.DataFrame(rows)
        payload = nearby_session_payload(frame.iloc[18], frame)
        self.assertIn(rows[19]["text"], payload["context"])
        self.assertNotIn(rows[0]["text"], payload["context"])
        self.assertLessEqual(len(payload["context"]), 8)

    def test_payloads_are_unique_by_request_text(self):
        frame = pd.DataFrame([
            dict(sample_id="1", system="s", timestamp=pd.Timestamp("2026-01-01"), group_id="g", text="GET /a HTTP/1.1"),
            dict(sample_id="2", system="s", timestamp=pd.Timestamp("2026-01-01 00:00:01"), group_id="g", text="GET /a HTTP/1.1"),
            dict(sample_id="3", system="s", timestamp=pd.Timestamp("2026-01-01 00:00:02"), group_id="g", text="GET /b HTTP/1.1"),
        ])
        payloads = build_payloads(frame, "b_to_a")
        self.assertEqual(len(payloads), 2)
        self.assertEqual(len({p["sample_id"] for p in payloads}), 2)

    def test_srbh_request_only_loading_preserves_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "srbh.csv"
            pd.DataFrame([
                {"timestamp": "17/Jul/2020:12:23:34 +0100", "request_http_method": "GET",
                 "request_http_request": "/", "request_http_protocol": "HTTP/1.1",
                 "000 - Normal": "1", "66 - SQL Injection": "0"},
                {"timestamp": "17/Jul/2020:12:23:35 +0100", "request_http_method": "GET",
                 "request_http_request": "/?id=1+union+select", "request_http_protocol": "HTTP/1.1",
                 "000 - Normal": "0", "66 - SQL Injection": "1"},
            ]).to_csv(path, index=False)
            requests = load_srbh(path, sample=1, seed=7, labelled=False)
            labelled = load_srbh(path, sample=1, seed=7, labelled=True)
            self.assertNotIn("label", requests.columns)
            self.assertIn("label", labelled.columns)
            self.assertEqual(requests.sample_id.tolist(), labelled.sample_id.tolist())
            self.assertEqual(requests.in_sample.tolist(), labelled.in_sample.tolist())

    def test_load_prepared_joins_labels_only_when_requested(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            pd.DataFrame([dict(sample_id="1", system="s", timestamp="2026-01-01",
                               group_id="g", text="GET / HTTP/1.1")]).to_csv(folder / "system_a.csv", index=False)
            pd.DataFrame([dict(sample_id="1", system="s", label="normal")]).to_csv(
                folder / "system_a_labels.csv", index=False)
            self.assertNotIn("label", load_prepared(folder, "system_a").columns)
            self.assertEqual(load_prepared(folder, "system_a", True).label.tolist(), ["normal"])

    def test_target_labels_are_absent_from_prompt_payload(self):
        frame = pd.DataFrame([dict(sample_id="1", system="s", timestamp=pd.Timestamp("2026-01-01"),
                                   group_id="g", text="GET / HTTP/1.1")])
        payload = build_payloads(frame, "b_to_a")[0]
        self.assertNotIn("label", json.dumps(payload))

    def test_custom_prompt_changes_cache_fingerprint(self):
        class Provider:
            def describe(self):
                return {"model": "test"}
        base = crossllm.Annotator(Provider(), [], None)
        changed = crossllm.Annotator(Provider(), [], None, system_prompt="custom")
        self.assertNotEqual(base.settings_hash, changed.settings_hash)

    def test_prompt_omits_empty_example_and_profile_sections(self):
        payload = {"candidate": "GET / HTTP/1.1", "context": []}
        shots = [{"label": "normal", "text": "GET /index HTTP/1.1"}]
        full = crossllm.render(shots, payload, profile=["GET /x HTTP/1.1"])[1]["content"]
        self.assertTrue(full.startswith(
            "LABELLED EXAMPLES FROM THE SOURCE SERVER:\n- [normal] GET /index HTTP/1.1\n\nMOST COMMON"))
        self.assertIn("- GET /x HTTP/1.1\n\nREQUEST UNDER REVIEW", full)
        bare = crossllm.render([], payload)[1]["content"]
        self.assertTrue(bare.startswith("REQUEST UNDER REVIEW"))
        self.assertIn("- (none)", bare)

    def test_metrics(self):
        result = metric_counts(["anomaly", "normal", "anomaly"],
                               ["anomaly", "anomaly", "normal"])
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 0))
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_row_output_propagates_one_text_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            target = pd.DataFrame([
                dict(sample_id="1", system="s", timestamp="2026-01-01", group_id="g", text="GET /a HTTP/1.1"),
                dict(sample_id="2", system="s", timestamp="2026-01-02", group_id="h", text="GET /a HTTP/1.1"),
            ])
            pd.DataFrame([dict(sample_id="1", system="s", label="normal"),
                          dict(sample_id="2", system="s", label="normal")]).to_csv(
                              folder / "system_a_labels.csv", index=False)
            record = {"sample_id": text_id("GET /a HTTP/1.1"), "status": "ok", "answer": {
                "decision": "normal", "confidence": .85, "reason_code": "normal_recovery",
                "short_reason": "ordinary request"}}
            summary = write_outputs(target, folder, "system_a", [record], folder)
            rows = pd.read_csv(folder / "row_predictions.csv")
            self.assertEqual(rows.predicted_label.tolist(), ["normal", "normal"])
            self.assertEqual(summary["coverage"]["labelled_rows"], 2)
            self.assertEqual(summary["coverage"]["failed_texts"], 0)
            self.assertEqual(summary["coverage"]["unselected_texts"], 0)


if __name__ == "__main__":
    unittest.main()
