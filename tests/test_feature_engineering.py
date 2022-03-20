"""
tests/test_feature_engineering.py

Unit tests for ml/feature_engineering.py and core/models.PairFeatures.

"""

import os
import sys
import unittest

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.models import CodeUnit, PairFeatures
from ml.feature_engineering import FeatureEngineer

EXPECTED_FEATURE_COUNT = 19



def _unit(uid: str, imports=None, annotations=None, domain_hints=None) -> CodeUnit:
    return CodeUnit(
        unit_id=uid,
        file_path=f"src/{uid}.py",
        language="python",
        imports=imports or [],
        annotations=annotations or [],
        domain_hints=domain_hints or [],
    )


def _engineer(
    units,
    edges=None,            # list of (u, v, weight, kind)
    cosine=None,
    semantic=None,
    centrality=None,
    behavioral=None,
    evolutionary=None,
) -> FeatureEngineer:
    n = len(units)
    g = nx.DiGraph()
    for u in units:
        g.add_node(u.unit_id)
    for (src, tgt, w, kind) in (edges or []):
        g.add_edge(src, tgt, weight=w, kind=kind)

    cos = cosine if cosine is not None else np.zeros((n, n))
    sem = semantic if semantic is not None else np.zeros((n, n))
    uid_idx = {u.unit_id: i for i, u in enumerate(units)}
    central = centrality or {}

    return FeatureEngineer(
        units=units,
        graph=g,
        cosine_sim=cos,
        semantic_sim=sem,
        centrality=central,
        unit_index=uid_idx,
        behavioral=behavioral,
        evolutionary=evolutionary,
    )



class TestPairFeaturesSchema(unittest.TestCase):

    def test_feature_names_count(self):
        self.assertEqual(len(PairFeatures.feature_names()), EXPECTED_FEATURE_COUNT)

    def test_feature_names_no_duplicates(self):
        names = PairFeatures.feature_names()
        self.assertEqual(len(names), len(set(names)))

    def test_to_feature_vector_length(self):
        pf = PairFeatures(comp_a="A", comp_b="B")
        self.assertEqual(len(pf.to_feature_vector()), EXPECTED_FEATURE_COUNT)

    def test_to_feature_vector_all_float(self):
        pf = PairFeatures(comp_a="A", comp_b="B",
                          shared_import_count=3,
                          inheritance_linked=1,
                          cross_layer_flag=1)
        for val in pf.to_feature_vector():
            self.assertIsInstance(val, float, f"Non-float value: {val!r}")

    def test_feature_names_matches_vector_order(self):
        """Each name should correspond to the matching position in the vector."""
        pf = PairFeatures(
            comp_a="A", comp_b="B",
            structural_coupling_weight=0.1,
            tfidf_cosine_similarity=0.2,
            execution_order_stability=0.5,
            change_sequence_directionality=0.7,
            cross_layer_flag=1,
        )
        vec = pf.to_feature_vector()
        names = PairFeatures.feature_names()
        idx_eos = names.index("execution_order_stability")
        idx_csd = names.index("change_sequence_directionality")
        idx_clf = names.index("cross_layer_flag")
        self.assertAlmostEqual(vec[idx_eos], 0.5)
        self.assertAlmostEqual(vec[idx_csd], 0.7)
        self.assertAlmostEqual(vec[idx_clf], 1.0)

    def test_new_behavioral_feature_in_names(self):
        self.assertIn("execution_order_stability", PairFeatures.feature_names())

    def test_new_evolutionary_feature_in_names(self):
        self.assertIn("change_sequence_directionality", PairFeatures.feature_names())



class TestSampleCandidates(unittest.TestCase):

    def test_graph_edge_included(self):
        units = [_unit("A"), _unit("B"), _unit("C")]
        eng = _engineer(units, edges=[("A", "B", 1, "call")])
        cands = eng._sample_candidates()
        self.assertIn(("A", "B"), cands)

    def test_behavioral_pair_included(self):
        units = [_unit("A"), _unit("B"), _unit("C")]
        eng = _engineer(units, behavioral={("B", "C"): {"runtime_call_frequency": 0.5}})
        cands = eng._sample_candidates()
        self.assertIn(("B", "C"), cands)

    def test_evolutionary_pair_included(self):
        units = [_unit("A"), _unit("B"), _unit("C")]
        eng = _engineer(units, evolutionary={("A", "C"): {"co_change_frequency": 0.1}})
        cands = eng._sample_candidates()
        self.assertIn(("A", "C"), cands)

    def test_no_self_pairs(self):
        units = [_unit("A"), _unit("B")]
        eng = _engineer(units)
        for (a, b) in eng._sample_candidates():
            self.assertNotEqual(a, b)

    def test_all_keys_are_canonical(self):
        units = [_unit("A"), _unit("B"), _unit("C")]
        eng = _engineer(units, edges=[("C", "A", 1, "call"), ("B", "A", 1, "call")])
        for (a, b) in eng._sample_candidates():
            self.assertEqual((a, b), (min(a, b), max(a, b)))

    def test_cosine_threshold_filters_low_similarity(self):
        units = [_unit("A"), _unit("B")]
        cos = np.array([[1.0, 0.01], [0.01, 1.0]])  # below 0.05 threshold
        eng = _engineer(units, cosine=cos)
        # No graph edges, no behavioral/evolutionary → should be empty
        cands = eng._sample_candidates()
        self.assertEqual(cands, set())

    def test_cosine_above_threshold_included(self):
        units = [_unit("A"), _unit("B")]
        cos = np.array([[1.0, 0.8], [0.8, 1.0]])
        eng = _engineer(units, cosine=cos)
        cands = eng._sample_candidates()
        self.assertIn(("A", "B"), cands)

    def test_unknown_unit_ids_excluded(self):
        units = [_unit("A"), _unit("B")]
        eng = _engineer(units, behavioral={("A", "Ghost"): {"runtime_call_frequency": 0.9}})
        cands = eng._sample_candidates()
        self.assertNotIn(("A", "Ghost"), cands)


class TestBuildOne(unittest.TestCase):

    def _base(self, units, **kwargs):
        return _engineer(units, **kwargs)

    def test_structural_coupling_weight_from_graph(self):
        units = [_unit("A"), _unit("B")]
        eng = self._base(units, edges=[("A", "B", 7, "call")])
        pf = eng._build_one("A", "B")
        self.assertAlmostEqual(pf.structural_coupling_weight, 7.0)

    def test_tfidf_and_semantic_from_matrices(self):
        units = [_unit("A"), _unit("B")]
        cos = np.array([[1.0, 0.6], [0.6, 1.0]])
        sem = np.array([[1.0, 0.4], [0.4, 1.0]])
        eng = self._base(units, cosine=cos, semantic=sem)
        pf = eng._build_one("A", "B")
        self.assertAlmostEqual(pf.tfidf_cosine_similarity, 0.6)
        self.assertAlmostEqual(pf.semantic_similarity, 0.4)

    def test_shared_imports(self):
        units = [
            _unit("A", imports=["os", "sys", "json"]),
            _unit("B", imports=["os", "sys", "re"]),
        ]
        eng = self._base(units)
        pf = eng._build_one("A", "B")
        self.assertEqual(pf.shared_import_count, 2)

    def test_shared_annotations(self):
        units = [
            _unit("A", annotations=["@Service", "@Transactional"]),
            _unit("B", annotations=["@Service", "@Repository"]),
        ]
        eng = self._base(units)
        pf = eng._build_one("A", "B")
        self.assertEqual(pf.shared_annotation_count, 1)

    def test_inheritance_linked(self):
        units = [_unit("A"), _unit("B")]
        eng = self._base(units, edges=[("A", "B", 1, "inheritance")])
        pf = eng._build_one("A", "B")
        self.assertEqual(pf.inheritance_linked, 1)

    def test_inheritance_not_linked_for_call(self):
        units = [_unit("A"), _unit("B")]
        eng = self._base(units, edges=[("A", "B", 1, "call")])
        pf = eng._build_one("A", "B")
        self.assertEqual(pf.inheritance_linked, 0)

    def test_pagerank_and_betweenness_from_centrality(self):
        units = [_unit("A"), _unit("B")]
        central = {
            "A": {"pagerank": 0.3, "betweenness": 0.1},
            "B": {"pagerank": 0.5, "betweenness": 0.2},
        }
        eng = self._base(units, centrality=central)
        pf = eng._build_one("A", "B")
        self.assertAlmostEqual(pf.pagerank_a, 0.3)
        self.assertAlmostEqual(pf.pagerank_b, 0.5)
        self.assertAlmostEqual(pf.betweenness_a, 0.1)
        self.assertAlmostEqual(pf.betweenness_b, 0.2)

    def test_cross_layer_flag_controller_repository(self):
        units = [
            _unit("A", domain_hints=["controller"]),
            _unit("B", domain_hints=["repository"]),
        ]
        eng = self._base(units)
        pf = eng._build_one("A", "B")
        self.assertEqual(pf.cross_layer_flag, 1)

    def test_cross_layer_flag_same_layer_zero(self):
        units = [
            _unit("A", domain_hints=["controller"]),
            _unit("B", domain_hints=["controller"]),
        ]
        eng = self._base(units)
        pf = eng._build_one("A", "B")
        self.assertEqual(pf.cross_layer_flag, 0)

    def test_behavioral_features_wired(self):
        units = [_unit("A"), _unit("B")]
        beh = {("A", "B"): {
            "runtime_call_frequency":    0.8,
            "runtime_call_depth":        3.0,
            "temporal_affinity":         0.6,
            "execution_order_stability": 0.9,
        }}
        eng = self._base(units, behavioral=beh)
        pf = eng._build_one("A", "B")
        self.assertAlmostEqual(pf.runtime_call_frequency, 0.8)
        self.assertAlmostEqual(pf.runtime_call_depth, 3.0)
        self.assertAlmostEqual(pf.temporal_affinity, 0.6)
        self.assertAlmostEqual(pf.execution_order_stability, 0.9)

    def test_evolutionary_features_wired(self):
        units = [_unit("A"), _unit("B")]
        evol = {("A", "B"): {
            "co_change_frequency":            0.3,
            "co_change_recency":              1.5,
            "logical_coupling_score":         0.7,
            "change_sequence_directionality": 0.85,
        }}
        eng = self._base(units, evolutionary=evol)
        pf = eng._build_one("A", "B")
        self.assertAlmostEqual(pf.co_change_frequency, 0.3)
        self.assertAlmostEqual(pf.co_change_recency, 1.5)
        self.assertAlmostEqual(pf.logical_coupling_score, 0.7)
        self.assertAlmostEqual(pf.change_sequence_directionality, 0.85)

    def test_behavioral_key_reverse_lookup(self):
        """Behavioral features stored under (B, A) should be picked up for build_one(A, B)."""
        units = [_unit("A"), _unit("B")]
        beh = {("B", "A"): {"runtime_call_frequency": 0.5, "runtime_call_depth": 1.0,
                             "temporal_affinity": 0.3, "execution_order_stability": 0.4}}
        eng = self._base(units, behavioral=beh)
        pf = eng._build_one("A", "B")
        self.assertAlmostEqual(pf.runtime_call_frequency, 0.5)

    def test_missing_behavioral_defaults_to_zero(self):
        units = [_unit("A"), _unit("B")]
        eng = self._base(units)
        pf = eng._build_one("A", "B")
        self.assertEqual(pf.runtime_call_frequency, 0.0)
        self.assertEqual(pf.execution_order_stability, 0.0)

    def test_missing_evolutionary_defaults_to_zero(self):
        units = [_unit("A"), _unit("B")]
        eng = self._base(units)
        pf = eng._build_one("A", "B")
        self.assertEqual(pf.co_change_frequency, 0.0)
        self.assertEqual(pf.change_sequence_directionality, 0.0)


class TestBuildPairsLabels(unittest.TestCase):

    def _eng_with_edge(self):
        units = [_unit("A"), _unit("B"), _unit("C")]
        return _engineer(units, edges=[("A", "B", 1, "call"), ("B", "C", 1, "call")])

    def test_label_assigned_from_canonical_key(self):
        eng = self._eng_with_edge()
        labels = {("A", "B"): 1, ("B", "C"): 0}
        pairs = eng.build_pairs(labelled_pairs=labels)
        ab = next((p for p in pairs if {p.comp_a, p.comp_b} == {"A", "B"}), None)
        self.assertIsNotNone(ab)
        self.assertEqual(ab.label, 1)

    def test_unlabelled_pair_has_none_label(self):
        eng = self._eng_with_edge()
        labels = {("A", "B"): 1}
        pairs = eng.build_pairs(labelled_pairs=labels)
        bc = next((p for p in pairs if {p.comp_a, p.comp_b} == {"B", "C"}), None)
        self.assertIsNotNone(bc)
        self.assertIsNone(bc.label)

    def test_no_labels_all_none(self):
        eng = self._eng_with_edge()
        pairs = eng.build_pairs()
        for p in pairs:
            self.assertIsNone(p.label)

    def test_build_pairs_returns_pair_features(self):
        eng = self._eng_with_edge()
        pairs = eng.build_pairs()
        self.assertTrue(all(isinstance(p, PairFeatures) for p in pairs))

    def test_build_pairs_vector_length(self):
        eng = self._eng_with_edge()
        for pf in eng.build_pairs():
            self.assertEqual(len(pf.to_feature_vector()), EXPECTED_FEATURE_COUNT)


if __name__ == "__main__":
    unittest.main()
