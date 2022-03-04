"""
tests/test_behavioral.py
Unit tests for signals/behavioral.py.
"""

import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from signals.behavioral import BehavioralSignalExtractor, _WINDOW_SECONDS


def _write_trace(content: str) -> str:
    """Write trace content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name



class TestResolve(unittest.TestCase):
    """_resolve maps log tokens to known unit_ids."""

    def setUp(self):
        self.ext = BehavioralSignalExtractor(
            ["com.example.OrderService", "com.example.PaymentService", "com.example.InventoryRepo"]
        )

    def test_exact_match(self):
        self.assertEqual(self.ext._resolve("com.example.OrderService"), "com.example.OrderService")

    def test_short_name_match(self):
        self.assertEqual(self.ext._resolve("OrderService"), "com.example.OrderService")

    def test_partial_suffix_match(self):
        self.assertEqual(self.ext._resolve("InventoryRepo"), "com.example.InventoryRepo")

    def test_unknown_returns_none(self):
        self.assertIsNone(self.ext._resolve("UnknownComponent"))


class TestCallNotationParsing(unittest.TestCase):
    """CALL A -> B [depth=N] format populates call_count and depth matrices."""

    def setUp(self):
        self.units = ["OrderService", "PaymentService", "InventoryRepo"]
        self.ext = BehavioralSignalExtractor(self.units)

    def test_single_call_recorded(self):
        trace = _write_trace("""\
            CALL OrderService -> PaymentService [depth=2]
        """)
        try:
            self.ext.load_trace_files([trace])
            i = self.ext._unit_index["OrderService"]
            j = self.ext._unit_index["PaymentService"]
            self.assertEqual(self.ext._call_count[i, j], 1.0)
            self.assertEqual(self.ext._call_depth[i, j], 2.0)
            self.assertEqual(self.ext._depth_count[i, j], 1.0)
        finally:
            os.unlink(trace)

    def test_multiple_calls_accumulate(self):
        lines = "\n".join(
            "CALL OrderService -> PaymentService [depth=1]" for _ in range(5)
        )
        trace = _write_trace(lines)
        try:
            self.ext.load_trace_files([trace])
            i = self.ext._unit_index["OrderService"]
            j = self.ext._unit_index["PaymentService"]
            self.assertEqual(self.ext._call_count[i, j], 5.0)
            self.assertAlmostEqual(self.ext._call_depth[i, j], 5.0)
        finally:
            os.unlink(trace)

    def test_depth_zero_when_omitted(self):
        trace = _write_trace("CALL OrderService -> InventoryRepo\n")
        try:
            self.ext.load_trace_files([trace])
            i = self.ext._unit_index["OrderService"]
            j = self.ext._unit_index["InventoryRepo"]
            self.assertEqual(self.ext._call_depth[i, j], 0.0)
        finally:
            os.unlink(trace)

    def test_reverse_direction_not_conflated(self):
        trace = _write_trace("""\
            CALL OrderService -> PaymentService [depth=1]
            CALL PaymentService -> OrderService [depth=1]
        """)
        try:
            self.ext.load_trace_files([trace])
            i = self.ext._unit_index["OrderService"]
            j = self.ext._unit_index["PaymentService"]
            self.assertEqual(self.ext._call_count[i, j], 1.0)
            self.assertEqual(self.ext._call_count[j, i], 1.0)
        finally:
            os.unlink(trace)


class TestTimestampedLogParsing(unittest.TestCase):
    """Python-log format populates co-occurrence and order_before matrices."""

    def setUp(self):
        self.units = ["OrderService", "PaymentService", "InventoryRepo"]
        self.ext = BehavioralSignalExtractor(self.units)

    def _log_line(self, ts: str, component: str) -> str:
        return f"{ts} - {component} - INFO doing work\n"

    def test_co_occurrence_within_window(self):
        trace = _write_trace(
            self._log_line("2024-01-01 10:00:00", "OrderService")
            + self._log_line("2024-01-01 10:00:03", "PaymentService")
        )
        try:
            self.ext.load_trace_files([trace])
            i = self.ext._unit_index["OrderService"]
            j = self.ext._unit_index["PaymentService"]
            self.assertGreater(self.ext._co_occurrence[i, j], 0)
        finally:
            os.unlink(trace)

    def test_no_co_occurrence_outside_window(self):
        gap = _WINDOW_SECONDS + 10
        trace = _write_trace(
            self._log_line("2024-01-01 10:00:00", "OrderService")
            + self._log_line(f"2024-01-01 10:00:{gap:02d}", "PaymentService")
        )
        try:
            self.ext.load_trace_files([trace])
            i = self.ext._unit_index["OrderService"]
            j = self.ext._unit_index["PaymentService"]
            self.assertEqual(self.ext._co_occurrence[i, j], 0)
        finally:
            os.unlink(trace)

    def test_order_before_recorded(self):
        # OrderService appears before PaymentService in the window
        trace = _write_trace(
            self._log_line("2024-01-01 10:00:00", "OrderService")
            + self._log_line("2024-01-01 10:00:02", "PaymentService")
        )
        try:
            self.ext.load_trace_files([trace])
            i = self.ext._unit_index["OrderService"]
            j = self.ext._unit_index["PaymentService"]
            # PaymentService arrived after OrderService → order_before[i, j] incremented
            self.assertGreater(self.ext._order_before[i, j], 0)
        finally:
            os.unlink(trace)


class TestGetPairFeatures(unittest.TestCase):
    """get_pair_features returns correctly computed feature dicts."""

    def setUp(self):
        self.units = ["A", "B", "C"]
        self.ext = BehavioralSignalExtractor(self.units)

    def _load(self, content: str):
        trace = _write_trace(content)
        self.ext.load_trace_files([trace])
        os.unlink(trace)

    def test_keys_are_canonical_min_max(self):
        self._load("CALL A -> B [depth=1]\n")
        features = self.ext.get_pair_features()
        for key in features:
            self.assertEqual(key, (min(key), max(key)), f"Key {key} not canonical")

    def test_feature_keys_present(self):
        self._load("CALL A -> B [depth=1]\n")
        features = self.ext.get_pair_features()
        self.assertTrue(features, "Expected at least one pair")
        for feat_dict in features.values():
            for name in ("runtime_call_frequency", "runtime_call_depth",
                         "temporal_affinity", "execution_order_stability"):
                self.assertIn(name, feat_dict, f"Missing feature: {name}")

    def test_runtime_call_frequency_normalized(self):
        # 3 calls A->B, 1 call A->C → total 4 calls
        self._load(
            "CALL A -> B [depth=1]\n"
            "CALL A -> B [depth=1]\n"
            "CALL A -> B [depth=1]\n"
            "CALL A -> C [depth=1]\n"
        )
        features = self.ext.get_pair_features()
        ab = features.get(("A", "B"), {})
        ac = features.get(("A", "C"), {})
        self.assertAlmostEqual(ab["runtime_call_frequency"], 3 / 4)
        self.assertAlmostEqual(ac["runtime_call_frequency"], 1 / 4)

    def test_runtime_call_depth_average(self):
        self._load(
            "CALL A -> B [depth=2]\n"
            "CALL A -> B [depth=4]\n"
        )
        features = self.ext.get_pair_features()
        ab = features.get(("A", "B"), {})
        self.assertAlmostEqual(ab["runtime_call_depth"], 3.0)

    def test_execution_order_stability_unidirectional(self):
        """Many A->B calls with no B->A: stability via call-direction asymmetry."""
        lines = "\n".join("CALL A -> B [depth=1]" for _ in range(6))
        self._load(lines + "\n")
        features = self.ext.get_pair_features()
        ab = features.get(("A", "B"), {})
        self.assertAlmostEqual(ab["execution_order_stability"], 1.0)

    def test_execution_order_stability_bidirectional(self):
        """Equal A->B and B->A calls → stability = 0.0."""
        lines = (
            "\n".join("CALL A -> B [depth=1]" for _ in range(3))
            + "\n"
            + "\n".join("CALL B -> A [depth=1]" for _ in range(3))
        )
        self._load(lines + "\n")
        features = self.ext.get_pair_features()
        ab = features.get(("A", "B"), {})
        self.assertAlmostEqual(ab["execution_order_stability"], 0.0)

    def test_execution_order_stability_from_timestamps(self):
        """With enough timestamped events, order_before drives stability."""
        lines = ""
        base = "2024-01-01 10:00:0"
        for sec in range(4):
            lines += f"{base}{sec} - A - INFO\n"
            lines += f"{base}{sec + 1} - B - INFO\n" if sec < 3 else f"2024-01-01 10:00:04 - B - INFO\n"
        self._load(lines)
        features = self.ext.get_pair_features()
        ab = features.get(("A", "B"), {})
        self.assertGreater(ab.get("execution_order_stability", 0.0), 0.0)

    def test_no_self_pairs(self):
        self._load("CALL A -> B [depth=1]\n")
        for key in self.ext.get_pair_features():
            self.assertNotEqual(key[0], key[1])

    def test_empty_traces_returns_empty(self):
        features = self.ext.get_pair_features()
        self.assertEqual(features, {})


class TestGetHotpaths(unittest.TestCase):
    """get_hotpaths returns top-k call pairs sorted by frequency."""

    def test_top_k_ordering(self):
        ext = BehavioralSignalExtractor(["A", "B", "C"])
        trace = _write_trace(
            "CALL A -> B [depth=1]\n" * 5
            + "CALL A -> C [depth=1]\n" * 2
        )
        ext.load_trace_files([trace])
        os.unlink(trace)

        hotpaths = ext.get_hotpaths(top_k=2)
        self.assertEqual(len(hotpaths), 2)
        self.assertEqual(hotpaths[0]["from"], "A")
        self.assertEqual(hotpaths[0]["to"], "B")
        self.assertEqual(hotpaths[0]["calls"], 5)

    def test_top_k_limit_respected(self):
        ext = BehavioralSignalExtractor(["A", "B", "C"])
        trace = _write_trace(
            "CALL A -> B [depth=1]\n"
            "CALL A -> C [depth=1]\n"
            "CALL B -> C [depth=1]\n"
        )
        ext.load_trace_files([trace])
        os.unlink(trace)
        self.assertLessEqual(len(ext.get_hotpaths(top_k=1)), 1)


class TestMissingTraceFile(unittest.TestCase):
    """load_trace_files should warn and skip missing files gracefully."""

    def test_missing_file_does_not_raise(self):
        ext = BehavioralSignalExtractor(["A", "B"])
        ext.load_trace_files(["/nonexistent/path/trace.log"])
        self.assertEqual(ext.get_pair_features(), {})


if __name__ == "__main__":
    unittest.main()
