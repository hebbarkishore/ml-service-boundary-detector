import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False
    logging.warning("imbalanced-learn not installed – SMOTE disabled.")

try:
    import hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False
    logging.warning("hdbscan not installed – clustering fallback disabled.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.models import PairFeatures, BoundaryCandidate, CodeUnit
from config import (
    GB_N_ESTIMATORS, GB_MAX_DEPTH, GB_LEARNING_RATE, GB_SUBSAMPLE,
    RANDOM_STATE, STRUCTURAL_WEIGHT, BEHAVIORAL_WEIGHT, EVOLUTIONARY_WEIGHT,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_METRIC,
    MODEL_DIR, ADAPTIVE_WEIGHT_ALPHA, ADAPTIVE_WEIGHT_MIN_FEEDBACK,
)

log = logging.getLogger(__name__)

_MODEL_PATH = MODEL_DIR / "boundary_ranker.joblib"


class BoundaryRanker:
    """
    Trains (optional) and applies the boundary ranking model.

    Usage - supervised
    ------------------
    ranker = BoundaryRanker()
    ranker.train(labelled_pairs)
    candidates = ranker.rank(all_pairs)

    Usage - unsupervised
    --------------------
    ranker = BoundaryRanker()
    candidates = ranker.rank_unsupervised(all_pairs)
    """

    def __init__(self):
        self._pipeline: Optional[Pipeline]  = None
        self._is_trained: bool              = False
        self._feature_importances: Dict     = {}


    def train(
        self,
        pairs: List[PairFeatures],
        cv_folds: int = 5,
    ) -> Dict:
        """
        Train a GradientBoostingClassifier on labelled PairFeatures.
        Returns a dict of training metrics.
        """
        labelled = [p for p in pairs if p.label is not None]
        if len(labelled) < 20:
            log.warning(
                "Only %d labelled samples available – supervised training skipped. "
                "Use rank_unsupervised() instead.", len(labelled)
            )
            return {"error": "insufficient_labels", "n_labelled": len(labelled)}

        X = np.array([p.to_feature_vector() for p in labelled], dtype=np.float32)
        y = np.array([p.label for p in labelled], dtype=int)

        log.info("Training on %d labelled pairs (pos=%d, neg=%d) …",
                 len(labelled), y.sum(), (y == 0).sum())

        if _HAS_SMOTE and y.sum() >= 3 and (y == 0).sum() >= 3:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(3, y.sum()-1))
            try:
                X, y = smote.fit_resample(X, y)
                log.info("After SMOTE: %d samples", len(y))
            except Exception as e:
                log.warning("SMOTE failed: %s", e)

        gb = GradientBoostingClassifier(
            n_estimators  = GB_N_ESTIMATORS,
            max_depth      = GB_MAX_DEPTH,
            learning_rate  = GB_LEARNING_RATE,
            subsample      = GB_SUBSAMPLE,
            random_state   = RANDOM_STATE,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        scaler = StandardScaler()

        self._pipeline = Pipeline([("scaler", scaler), ("gb", gb)])
        self._pipeline.fit(X, y)
        self._is_trained = True

        importances = gb.feature_importances_
        feat_names  = PairFeatures.feature_names()
        self._feature_importances = dict(
            sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
        )

        metrics = {}
        if len(set(y)) >= 2 and len(y) >= cv_folds * 2:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            try:
                auc_scores = cross_val_score(
                    Pipeline([("scaler", StandardScaler()), ("gb",
                        GradientBoostingClassifier(
                            n_estimators=GB_N_ESTIMATORS,
                            max_depth=GB_MAX_DEPTH,
                            learning_rate=GB_LEARNING_RATE,
                            random_state=RANDOM_STATE,
                        )
                    )]),
                    X, y, cv=cv, scoring="roc_auc", n_jobs=-1,
                )
                metrics["cv_roc_auc_mean"] = float(auc_scores.mean())
                metrics["cv_roc_auc_std"]  = float(auc_scores.std())
                log.info("CV ROC-AUC: %.3f ± %.3f",
                         metrics["cv_roc_auc_mean"], metrics["cv_roc_auc_std"])
            except Exception as e:
                log.warning("Cross-validation failed: %s", e)

        metrics["n_features"]       = len(feat_names)
        metrics["n_train_samples"]  = len(y)
        metrics["feature_importances"] = self._feature_importances

        # Persist
        joblib.dump(self._pipeline, _MODEL_PATH)
        log.info("Model saved to %s", _MODEL_PATH)

        return metrics

    def load(self) -> bool:
        """Load a previously saved model."""
        if _MODEL_PATH.exists():
            self._pipeline   = joblib.load(_MODEL_PATH)
            self._is_trained = True
            log.info("Loaded existing model from %s", _MODEL_PATH)
            return True
        return False


    def rank(self, pairs: List[PairFeatures]) -> List[BoundaryCandidate]:
        """
        Score all pairs with the trained model and return ranked candidates.
        Falls back to unsupervised if model is not trained.
        """
        if not self._is_trained:
            if not self.load():
                log.warning("No trained model – falling back to unsupervised ranking.")
                return self.rank_unsupervised(pairs)

        X = np.array([p.to_feature_vector() for p in pairs], dtype=np.float32)
        probs = self._pipeline.predict_proba(X)[:, 1]

        candidates = []
        for pair, prob in zip(pairs, probs):
            rationale = self._build_rationale(pair)
            candidates.append(BoundaryCandidate(
                comp_a          = pair.comp_a,
                comp_b          = pair.comp_b,
                boundary_score  = float(prob),
                confidence      = float(prob),
                rationale       = rationale,
                suggested_service = self._suggest_name(pair),
            ))

        return sorted(candidates, key=lambda c: c.boundary_score, reverse=True)



    def rank_unsupervised(
        self,
        pairs: List[PairFeatures],
        feedback_labels: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> List[BoundaryCandidate]:
        """
        Compute a weighted composite score without a trained supervised model.
        Higher score = stronger boundary signal (components should be separated).

        When `feedback_labels` contains enough architect decisions the channel
        weights are shifted toward whichever signal best discriminates accepted
        from rejected boundaries (adaptive weighting).  The base config weights
        are always used as a prior so the result degrades gracefully to the
        original behaviour when no feedback is available.
        """

        def safe_norm(vals):
            arr = np.array(vals, dtype=np.float32)
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / rng if rng > 1e-9 else arr * 0

        struct_raw = [
            p.structural_coupling_weight * 0.4 +
            (1.0 - p.tfidf_cosine_similarity) * 0.4 +
            (1.0 - p.semantic_similarity) * 0.2
            for p in pairs
        ]
        struct_norm = safe_norm(struct_raw)

        behav_raw = [
            p.runtime_call_frequency * 0.5 +
            p.temporal_affinity * 0.3 +
            p.execution_order_stability * 0.2
            for p in pairs
        ]
        behav_norm = safe_norm(behav_raw)

        evol_raw = [
            (1.0 - p.logical_coupling_score) * 0.4 +
            (1.0 - p.co_change_frequency) * 0.3 +
            p.change_sequence_directionality * 0.3
            for p in pairs
        ]
        evol_norm = safe_norm(evol_raw)

        w_s, w_b, w_e, adapted = self._compute_adaptive_weights(
            pairs, struct_norm, behav_norm, evol_norm, feedback_labels or {}
        )

        candidates = []
        for i, pair in enumerate(pairs):
            score = (
                w_s * float(struct_norm[i]) +
                w_b * float(behav_norm[i]) +
                w_e * float(evol_norm[i])
            )
            if pair.cross_layer_flag:
                score = min(1.0, score + 0.15)

            rationale = {
                "structural_channel":   round(float(struct_norm[i]) * w_s, 5),
                "behavioral_channel":   round(float(behav_norm[i])  * w_b, 5),
                "evolutionary_channel": round(float(evol_norm[i])   * w_e, 5),
                "cross_layer_bonus":    round(float(pair.cross_layer_flag) * 0.15, 5),
                "weights_adapted":      adapted,
                "w_structural":         round(w_s, 4),
                "w_behavioral":         round(w_b, 4),
                "w_evolutionary":       round(w_e, 4),
            }
            candidates.append(BoundaryCandidate(
                comp_a            = pair.comp_a,
                comp_b            = pair.comp_b,
                boundary_score    = score,
                confidence        = score,
                rationale         = rationale,
                suggested_service = self._suggest_name(pair),
            ))

        return sorted(candidates, key=lambda c: c.boundary_score, reverse=True)

    def _compute_adaptive_weights(
        self,
        pairs:       List[PairFeatures],
        struct_norm: np.ndarray,
        behav_norm:  np.ndarray,
        evol_norm:   np.ndarray,
        feedback_labels: Dict[Tuple[str, str], int],
    ) -> Tuple[float, float, float, bool]:
        """
        Shift the three channel weights toward whichever signal best separates
        architect-accepted from architect-rejected boundaries.

        Algorithm
        ---------
        1. For each pair that has a feedback label, record its normalised
           sub-scores and its label (1 = boundary, 0 = keep together).
        2. Compute per-channel discrimination power:
               disc_c = |mean_c(accepted) - mean_c(rejected)|
        3. Blend base weights with discrimination-driven weights using ALPHA:
               w_c = base_c * (1 + ALPHA * disc_c / sum(disc))
        4. Re-normalise so the three weights still sum to 1.

        Returns (w_structural, w_behavioral, w_evolutionary, was_adapted).
        Falls back to base config weights when feedback is insufficient.
        """
        base = (STRUCTURAL_WEIGHT, BEHAVIORAL_WEIGHT, EVOLUTIONARY_WEIGHT)

        if len(feedback_labels) < ADAPTIVE_WEIGHT_MIN_FEEDBACK:
            return (*base, False)

        pair_key = {
            (min(p.comp_a, p.comp_b), max(p.comp_a, p.comp_b)): idx
            for idx, p in enumerate(pairs)
        }

        scores_by_label: Dict[int, List[Tuple[float, float, float]]] = {0: [], 1: []}
        for (a, b), label in feedback_labels.items():
            key = (min(a, b), max(a, b))
            idx = pair_key.get(key)
            if idx is None or label not in (0, 1):
                continue
            scores_by_label[label].append((
                float(struct_norm[idx]),
                float(behav_norm[idx]),
                float(evol_norm[idx]),
            ))

        accepted = scores_by_label[1]
        rejected = scores_by_label[0]

        if not accepted or not rejected:
            log.debug("Adaptive weights: missing accepted or rejected feedback — using base weights.")
            return (*base, False)

        def col_mean(rows, col):
            return sum(r[col] for r in rows) / len(rows)

        disc = [
            abs(col_mean(accepted, c) - col_mean(rejected, c))
            for c in range(3)
        ]
        disc_total = sum(disc) or 1.0

        alpha = ADAPTIVE_WEIGHT_ALPHA
        raw = [
            base[c] * (1.0 + alpha * disc[c] / disc_total)
            for c in range(3)
        ]
        total = sum(raw)
        w_s, w_b, w_e = (r / total for r in raw)

        log.info(
            "Adaptive weights from %d feedback pairs  "
            "→ structural=%.3f  behavioral=%.3f  evolutionary=%.3f",
            len(accepted) + len(rejected), w_s, w_b, w_e,
        )
        return w_s, w_b, w_e, True



    def suggest_clusters(
        self,
        units: List[CodeUnit],
        tfidf_matrix,               # scipy sparse matrix
    ) -> Dict[int, List[str]]:
        """
        Uses HDBSCAN on TF-IDF vectors to suggest initial service groupings.
        Cluster label -1 = noise (singletons that don't belong to any group).
        Returns {cluster_id: [unit_id, …]}
        """
        if not _HAS_HDBSCAN:
            log.warning("hdbscan not available – cluster suggestions skipped.")
            return {}

        try:
            from sklearn.decomposition import TruncatedSVD
            n_components = min(50, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
            if n_components < 2:
                return {}
            svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
            reduced = svd.fit_transform(tfidf_matrix)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(reduced)
        except Exception as e:
            log.error("HDBSCAN clustering failed: %s", e)
            return {}

        clusters: Dict[int, List[str]] = {}
        for uid, label in zip([u.unit_id for u in units], labels):
            clusters.setdefault(int(label), []).append(uid)
        return clusters



    def _build_rationale(self, pair: PairFeatures) -> Dict[str, float]:
        if not self._feature_importances:
            return {}
        vec  = pair.to_feature_vector()
        names = PairFeatures.feature_names()
        contributions = {}
        total_imp = sum(self._feature_importances.values()) or 1.0
        for name, val in zip(names, vec):
            imp = self._feature_importances.get(name, 0.0)
            contributions[name] = round(val * imp / total_imp, 5)
        return dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8])

    @staticmethod
    def _suggest_name(pair: PairFeatures) -> str:
        """
        Heuristically derives a candidate service name from the component pair.
        Strips common suffixes and uses the longer, more domain-specific segment.
        """
        def clean(uid: str) -> str:
            parts   = uid.split(".")
            name    = parts[-1] if parts else uid
            for suffix in ("Service", "Controller", "Repository", "Handler",
                           "Manager", "Facade", "Impl", "Dao", "Repo"):
                name = name.replace(suffix, "")
            return name.strip() or uid

        a_clean = clean(pair.comp_a)
        b_clean = clean(pair.comp_b)
        # Return the more specific (longer) one
        candidate = a_clean if len(a_clean) >= len(b_clean) else b_clean
        return f"{candidate}Service" if candidate else "UnnamedService"

    @property
    def feature_importances(self) -> Dict[str, float]:
        return dict(self._feature_importances)
