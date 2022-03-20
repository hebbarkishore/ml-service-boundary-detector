"""
tests/test_evolutionary.py

Unit tests for signals/evolutionary.py.
"""

import os
import sys
import unittest
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from signals.evolutionary import EvolutionarySignalExtractor
from core.models import CodeUnit

# Module-level alias so tests can call the staticmethod without accidentally
_sequence_directionality = EvolutionarySignalExtractor._sequence_directionality




def _make_unit(uid: str, file_path: str) -> CodeUnit:
    return CodeUnit(unit_id=uid, file_path=file_path, language="python")


def _extractor_with_state(
    file_change_count: dict,
    co_change_raw: dict,
    co_change_decay: dict,
    file_commit_times: Optional[dict] = None,
) -> EvolutionarySignalExtractor:
    """Return an extractor with pre-populated internal state, skipping mine()."""
    ext = EvolutionarySignalExtractor.__new__(EvolutionarySignalExtractor)
    ext._repo_path = "."
    ext._repo = None
    ext._file_change_count = defaultdict(int, file_change_count)
    ext._co_change_raw = defaultdict(int, co_change_raw)
    ext._co_change_decay = defaultdict(float, co_change_decay)
    ext._file_commit_times = defaultdict(list, file_commit_times or {})
    ext._mined = True
    return ext



class TestNormalise(unittest.TestCase):

    def test_strips_leading_dotslash(self):
        result = EvolutionarySignalExtractor._normalise("./src/Foo.java")
        self.assertFalse(result.startswith("./"))

    def test_plain_path_unchanged(self):
        result = EvolutionarySignalExtractor._normalise("src/main/Foo.java")
        self.assertEqual(result, "src/main/Foo.java")

    def test_produces_string(self):
        result = EvolutionarySignalExtractor._normalise("src/main/Foo.java")
        self.assertIsInstance(result, str)




_DAY = 86400.0  # seconds in one day


class TestSequenceDirectionality(unittest.TestCase):

    def test_empty_times_returns_zero(self):
        self.assertEqual(_sequence_directionality([], [1.0, 2.0]), 0.0)
        self.assertEqual(_sequence_directionality([1.0], []), 0.0)

    def test_perfectly_unidirectional_a_leads(self):
        # A changes at t=0; B changes at t=1 day (within 7-day window)
        # B never leads A, so result should be 1.0
        times_a = sorted([0.0, _DAY * 10, _DAY * 20])
        times_b = sorted([_DAY * 1, _DAY * 11, _DAY * 21])
        result = _sequence_directionality(times_a, times_b)
        self.assertAlmostEqual(result, 1.0)

    def test_perfectly_unidirectional_b_leads(self):
        times_b = sorted([0.0, _DAY * 10, _DAY * 20])
        times_a = sorted([_DAY * 1, _DAY * 11, _DAY * 21])
        result = _sequence_directionality(times_a, times_b)
        self.assertAlmostEqual(result, 1.0)

    def test_result_in_range_for_interleaved(self):
        # Alternating A and B changes within the window — result between 0 and 1
        times_a = [0.0, _DAY * 2]
        times_b = [_DAY, _DAY * 3]
        result = _sequence_directionality(times_a, times_b)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_outside_window_not_counted(self):
        # A changes; B changes 30 days later (outside 7-day window)
        times_a = [0.0]
        times_b = [_DAY * 30]
        result = _sequence_directionality(times_a, times_b)
        self.assertEqual(result, 0.0)

    def test_result_in_range_random(self):
        import random
        random.seed(0)
        times_a = sorted(random.uniform(0, 1e6) for _ in range(20))
        times_b = sorted(random.uniform(0, 1e6) for _ in range(20))
        result = _sequence_directionality(times_a, times_b)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)



class TestGetPairFeatures(unittest.TestCase):

    def _make_units(self):
        return [
            _make_unit("OrderService",   "src/OrderService.py"),
            _make_unit("PaymentService", "src/PaymentService.py"),
            _make_unit("InventoryRepo",  "src/InventoryRepo.py"),
        ]

    def test_feature_keys_present(self):
        ext = _extractor_with_state(
            file_change_count={"src/OrderService.py": 5, "src/PaymentService.py": 4},
            co_change_raw={("src/OrderService.py", "src/PaymentService.py"): 4},
            co_change_decay={("src/OrderService.py", "src/PaymentService.py"): 2.0},
        )
        result = ext.get_pair_features(self._make_units())
        self.assertEqual(len(result), 1)
        feat = next(iter(result.values()))
        for key in ("co_change_frequency", "co_change_recency",
                    "logical_coupling_score", "change_sequence_directionality"):
            self.assertIn(key, feat, f"Missing key: {key}")

    def test_below_min_count_filtered(self):
        """Pairs with co_change_raw < CO_CHANGE_MIN_COUNT are excluded."""
        from config import CO_CHANGE_MIN_COUNT
        ext = _extractor_with_state(
            file_change_count={"src/OrderService.py": 5, "src/PaymentService.py": 4},
            co_change_raw={
                ("src/OrderService.py", "src/PaymentService.py"): CO_CHANGE_MIN_COUNT - 1
            },
            co_change_decay={},
        )
        result = ext.get_pair_features(self._make_units())
        self.assertEqual(result, {})

    def test_at_min_count_included(self):
        from config import CO_CHANGE_MIN_COUNT
        ext = _extractor_with_state(
            file_change_count={"src/OrderService.py": 5, "src/PaymentService.py": 5},
            co_change_raw={
                ("src/OrderService.py", "src/PaymentService.py"): CO_CHANGE_MIN_COUNT
            },
            co_change_decay={("src/OrderService.py", "src/PaymentService.py"): 1.0},
        )
        result = ext.get_pair_features(self._make_units())
        self.assertEqual(len(result), 1)

    def test_keys_are_canonical(self):
        ext = _extractor_with_state(
            file_change_count={"src/OrderService.py": 5, "src/PaymentService.py": 4},
            co_change_raw={("src/OrderService.py", "src/PaymentService.py"): 4},
            co_change_decay={("src/OrderService.py", "src/PaymentService.py"): 1.0},
        )
        result = ext.get_pair_features(self._make_units())
        for key in result:
            self.assertEqual(key, (min(key), max(key)))

    def test_co_change_frequency_normalized(self):
        # 4 co-changes; total change count = 5+4 = 9
        ext = _extractor_with_state(
            file_change_count={"src/OrderService.py": 5, "src/PaymentService.py": 4},
            co_change_raw={("src/OrderService.py", "src/PaymentService.py"): 4},
            co_change_decay={("src/OrderService.py", "src/PaymentService.py"): 1.0},
        )
        result = ext.get_pair_features(self._make_units())
        feat = next(iter(result.values()))
        self.assertAlmostEqual(feat["co_change_frequency"], 4 / 9)

    def test_directionality_wired_from_commit_times(self):
        """commit times for A always before B → directionality > 0."""
        DAY = 86400.0
        times_a = [0.0, DAY * 10, DAY * 20]
        times_b = [DAY * 1, DAY * 11, DAY * 21]
        ext = _extractor_with_state(
            file_change_count={"src/OrderService.py": 3, "src/PaymentService.py": 3},
            co_change_raw={("src/OrderService.py", "src/PaymentService.py"): 3},
            co_change_decay={("src/OrderService.py", "src/PaymentService.py"): 1.5},
            file_commit_times={
                "src/OrderService.py": times_a,
                "src/PaymentService.py": times_b,
            },
        )
        result = ext.get_pair_features(self._make_units())
        feat = next(iter(result.values()))
        self.assertGreater(feat["change_sequence_directionality"], 0.0)

    def test_unknown_files_excluded(self):
        """Files not mapped to any CodeUnit produce no pairs."""
        ext = _extractor_with_state(
            file_change_count={"unknown/Ghost.py": 10, "also/Unknown.py": 8},
            co_change_raw={("unknown/Ghost.py", "also/Unknown.py"): 5},
            co_change_decay={("unknown/Ghost.py", "also/Unknown.py"): 2.0},
        )
        result = ext.get_pair_features(self._make_units())
        self.assertEqual(result, {})



class TestGetChangeHotspots(unittest.TestCase):

    def _units(self):
        return [
            _make_unit("A", "src/A.py"),
            _make_unit("B", "src/B.py"),
            _make_unit("C", "src/C.py"),
        ]

    def test_ordered_by_change_count(self):
        ext = _extractor_with_state(
            file_change_count={"src/A.py": 10, "src/B.py": 3, "src/C.py": 7},
            co_change_raw={}, co_change_decay={},
        )
        hotspots = ext.get_change_hotspots(self._units(), top_k=3)
        counts = [h["change_count"] for h in hotspots]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_top_k_limit(self):
        ext = _extractor_with_state(
            file_change_count={"src/A.py": 10, "src/B.py": 3, "src/C.py": 7},
            co_change_raw={}, co_change_decay={},
        )
        self.assertEqual(len(ext.get_change_hotspots(self._units(), top_k=2)), 2)

    def test_unknown_files_ignored(self):
        ext = _extractor_with_state(
            file_change_count={"src/A.py": 5, "src/Ghost.py": 100},
            co_change_raw={}, co_change_decay={},
        )
        hotspots = ext.get_change_hotspots(self._units())
        unit_ids = [h["unit_id"] for h in hotspots]
        self.assertNotIn("Ghost", unit_ids)


class TestNoGitRepo(unittest.TestCase):

    def test_invalid_repo_path_does_not_raise(self):
        ext = EvolutionarySignalExtractor("/nonexistent/path/repo")
        ext.mine()  # should warn and set _mined=True without crashing
        self.assertTrue(ext._mined)
        units = [_make_unit("A", "src/A.py")]
        result = ext.get_pair_features(units)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
