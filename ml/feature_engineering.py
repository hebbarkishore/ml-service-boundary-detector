

import logging
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.models import CodeUnit, PairFeatures
from config import MAX_PAIRS_PER_COMPONENT

log = logging.getLogger(__name__)

# Layers considered "distant" – a direct edge between them with no intermediary
# is a cross-layer coupling smell
_DISTANT_PAIRS = {
    frozenset({"controller", "repository"}),
    frozenset({"controller", "model"}),
}


class FeatureEngineer:
    """
    Builds the full PairFeatures dataset from pre-computed signal outputs.
    """

    def __init__(
        self,
        units:        List[CodeUnit],
        graph:        nx.DiGraph,
        cosine_sim:   np.ndarray,          # n×n TF-IDF cosine
        semantic_sim: np.ndarray,          # n×n Word2Vec cosine
        centrality:   Dict[str, Dict],
        unit_index:   Dict[str, int],
        behavioral:   Optional[Dict[Tuple, Dict]] = None,
        evolutionary: Optional[Dict[Tuple, Dict]] = None,
    ):
        self._units      = units
        self._graph      = graph
        self._cos        = cosine_sim
        self._sem        = semantic_sim
        self._central    = centrality
        self._uid2idx    = unit_index
        self._behavioral = behavioral or {}
        self._evol       = evolutionary or {}

    # ── Public ────────────────────────────────────────────────────────────────

    def build_pairs(
        self,
        labelled_pairs: Optional[Dict[Tuple[str,str], int]] = None,
    ) -> List[PairFeatures]:
        """
        Build PairFeatures for all sampled pairs.
        labelled_pairs – optional {(a,b): label} ground-truth dict for supervised mode.
        """
        candidate_set = self._sample_candidates()
        log.info("Building features for %d candidate pairs …", len(candidate_set))

        result = []
        for (a, b) in candidate_set:
            pf = self._build_one(a, b)
            if labelled_pairs:
                key = (min(a, b), max(a, b))
                pf.label = labelled_pairs.get(key, labelled_pairs.get((a,b)))
            result.append(pf)

        log.info("Built %d PairFeatures objects.", len(result))
        return result

    # ── Pair sampling ─────────────────────────────────────────────────────────

    def _sample_candidates(self) -> Set[Tuple[str, str]]:
        candidates: Set[Tuple[str, str]] = set()
        n = len(self._units)
        uid_list = [u.unit_id for u in self._units]

        # (a) Graph neighbours
        for u, v in self._graph.edges():
            if u in self._uid2idx and v in self._uid2idx:
                candidates.add((min(u,v), max(u,v)))

        # (b) Top-K TF-IDF similar pairs per component
        k = min(MAX_PAIRS_PER_COMPONENT, n - 1)
        if self._cos.shape[0] == n:
            for i in range(n):
                top_j = np.argsort(self._cos[i])[::-1][1:k+1]
                for j in top_j:
                    if self._cos[i, j] > 0.05:   # ignore near-zero similarity
                        a, b = uid_list[i], uid_list[j]
                        candidates.add((min(a,b), max(a,b)))

        # (c) Pairs in behavioral signal
        for (a, b) in self._behavioral:
            candidates.add((min(a,b), max(a,b)))

        # (d) Pairs in evolutionary signal
        for (a, b) in self._evol:
            candidates.add((min(a,b), max(a,b)))

        # Filter: both endpoints must be known units
        uid_set = {u.unit_id for u in self._units}
        return {(a,b) for (a,b) in candidates if a in uid_set and b in uid_set and a != b}

    # ── Feature construction for a single pair ────────────────────────────────

    def _build_one(self, a: str, b: str) -> PairFeatures:
        ia = self._uid2idx.get(a, -1)
        ib = self._uid2idx.get(b, -1)
        ua = next((u for u in self._units if u.unit_id == a), None)
        ub = next((u for u in self._units if u.unit_id == b), None)

        pf = PairFeatures(comp_a=a, comp_b=b)

        # ── Structural ────────────────────────────────────────────────────────
        # Edge weight (bidirectional max)
        w_ab = self._graph[a][b]["weight"] if self._graph.has_edge(a, b) else 0
        w_ba = self._graph[b][a]["weight"] if self._graph.has_edge(b, a) else 0
        pf.structural_coupling_weight = float(max(w_ab, w_ba))

        # TF-IDF cosine similarity
        if ia >= 0 and ib >= 0 and self._cos.shape[0] > max(ia, ib):
            pf.tfidf_cosine_similarity = float(self._cos[ia, ib])

        # Semantic (Word2Vec) similarity
        if ia >= 0 and ib >= 0 and self._sem.shape[0] > max(ia, ib):
            pf.semantic_similarity = float(self._sem[ia, ib])

        # Shared imports
        if ua and ub:
            set_a = set(ua.imports)
            set_b = set(ub.imports)
            pf.shared_import_count    = len(set_a & set_b)
            pf.shared_annotation_count = len(set(ua.annotations) & set(ub.annotations))

        # Inheritance edge
        pf.inheritance_linked = int(
            self._graph.has_edge(a, b) and
            self._graph[a][b].get("kind", "") == "inheritance"
        )

        # Centrality
        pf.pagerank_a    = self._central.get(a, {}).get("pagerank", 0.0)
        pf.pagerank_b    = self._central.get(b, {}).get("pagerank", 0.0)
        pf.betweenness_a = self._central.get(a, {}).get("betweenness", 0.0)
        pf.betweenness_b = self._central.get(b, {}).get("betweenness", 0.0)

        # Cross-layer flag
        if ua and ub:
            layers_a = set(ua.domain_hints)
            layers_b = set(ub.domain_hints)
            for pair in _DISTANT_PAIRS:
                if layers_a & pair and layers_b & pair and layers_a != layers_b:
                    pf.cross_layer_flag = 1
                    break

        # ── Behavioral ────────────────────────────────────────────────────────
        beh_key = (min(a,b), max(a,b))
        beh_fwd = self._behavioral.get((a,b), {})
        beh_rev = self._behavioral.get((b,a), {})
        beh     = beh_fwd or beh_rev or self._behavioral.get(beh_key, {})
        if beh:
            pf.runtime_call_frequency = beh.get("runtime_call_frequency", 0.0)
            pf.runtime_call_depth     = beh.get("runtime_call_depth",     0.0)
            pf.temporal_affinity      = beh.get("temporal_affinity",      0.0)

        # ── Evolutionary ──────────────────────────────────────────────────────
        evol_key = (min(a,b), max(a,b))
        evol     = self._evol.get(evol_key, self._evol.get((a,b), self._evol.get((b,a), {})))
        if evol:
            pf.co_change_frequency    = evol.get("co_change_frequency",   0.0)
            pf.co_change_recency      = evol.get("co_change_recency",     0.0)
            pf.logical_coupling_score = evol.get("logical_coupling_score",0.0)

        return pf
