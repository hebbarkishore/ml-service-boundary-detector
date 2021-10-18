import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_DIR
from core.models import BoundaryCandidate, CodeUnit, PairFeatures
from signals.structural import StructuralSignalExtractor
from signals.behavioral import BehavioralSignalExtractor
from signals.evolutionary import EvolutionarySignalExtractor
from ingestion.document_ingester import DocumentIngester
from ml.feature_engineering import FeatureEngineer
from ml.boundary_ranker import BoundaryRanker

log = logging.getLogger(__name__)



@dataclass
class PipelineConfig:
    
    code_roots:      List[str]

    trace_files:     List[str]  = None

    doc_files:       List[str]  = None

    git_repo_path:   Optional[str] = None

    labels_file:     Optional[str] = None

    top_k:           int = 30

    skip_behavioral:  bool = False
    skip_evolutionary:bool = False
    skip_documents:   bool = False



@dataclass
class PipelineResult:
    units:             List[CodeUnit]
    top_candidates:    List[BoundaryCandidate]
    cluster_map:       Dict[int, List[str]]
    feature_importances: Dict[str, float]
    hotspots:          Dict
    meta:              Dict



class ServiceBoundaryPipeline:

    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def run(self) -> PipelineResult:
        t0 = time.time()
        cfg = self.cfg
        meta: Dict = {"stages": {}}

        # ── Stage 1: Structural ──────────────────────────────────────────────
        log.info("=== Stage 1: Structural Signal Extraction ===")
        t_s = time.time()
        structural_extractor = StructuralSignalExtractor()

        doc_corpus: List[str] = []
        if not cfg.skip_documents:
            log.info("=== Stage 2: Document Ingestion ===")
            ingester   = DocumentIngester()
            doc_chunks = ingester.ingest_folder()
            if cfg.doc_files:
                doc_chunks += ingester.ingest_files(cfg.doc_files)
            doc_corpus  = ingester.get_corpus_texts(doc_chunks)
            log.info("Ingested %d document chunks", len(doc_chunks))
            meta["stages"]["documents"] = {"chunks": len(doc_chunks)}

        structural = structural_extractor.extract(
            roots=cfg.code_roots,
            doc_corpus=doc_corpus or None,
        )
        units      = structural["units"]
        graph      = structural["graph"]
        unit_index = structural["unit_index"]
        meta["stages"]["structural"] = {
            "units": len(units),
            "edges": graph.number_of_edges(),
            "elapsed_s": round(time.time() - t_s, 2),
        }
        log.info("Structural: %d units, %d edges", len(units), graph.number_of_edges())

        if not units:
            log.error("No code units found. Check code_roots: %s", cfg.code_roots)
            return PipelineResult([], [], {}, {}, {}, meta)

        uid_list = [u.unit_id for u in units]

    
        behavioral_features = {}
        if not cfg.skip_behavioral and cfg.trace_files:
            log.info("=== Stage 3: Behavioral Signal Extraction ===")
            t_s = time.time()
            beh = BehavioralSignalExtractor(uid_list)
            beh.load_trace_files(cfg.trace_files)
            behavioral_features = beh.get_pair_features()
            meta["stages"]["behavioral"] = {
                "trace_files": len(cfg.trace_files),
                "pair_signals": len(behavioral_features),
                "elapsed_s": round(time.time() - t_s, 2),
            }
            log.info("Behavioral: %d pair signals extracted", len(behavioral_features))

    
        evolutionary_features = {}
        if not cfg.skip_evolutionary:
            log.info("=== Stage 4: Evolutionary Signal Extraction ===")
            t_s = time.time()
            repo_path = cfg.git_repo_path or cfg.code_roots[0]
            evo = EvolutionarySignalExtractor(repo_path)
            evo.mine()
            evolutionary_features = evo.get_pair_features(units)
            hotspots = {
                "change_hotspots": evo.get_change_hotspots(units),
                "recent_commits":  evo.get_commit_metadata(max_commits=10),
            }
            meta["stages"]["evolutionary"] = {
                "pair_signals": len(evolutionary_features),
                "elapsed_s":    round(time.time() - t_s, 2),
            }
            log.info("Evolutionary: %d co-change pairs", len(evolutionary_features))
        else:
            hotspots = {}

        log.info("=== Stage 5: Feature Engineering ===")
        t_s = time.time()
        engineer = FeatureEngineer(
            units        = units,
            graph        = graph,
            cosine_sim   = structural["cosine_sim"],
            semantic_sim = structural["semantic_sim"],
            centrality   = structural["centrality"],
            unit_index   = unit_index,
            behavioral   = behavioral_features,
            evolutionary = evolutionary_features,
        )

        labelled_pairs = self._load_labels(cfg.labels_file) if cfg.labels_file else None
        pairs = engineer.build_pairs(labelled_pairs)
        meta["stages"]["features"] = {
            "candidate_pairs": len(pairs),
            "elapsed_s": round(time.time() - t_s, 2),
        }
        log.info("Built %d candidate pairs", len(pairs))

        log.info("=== Stage 6: ML Ranking ===")
        t_s = time.time()
        ranker = BoundaryRanker()
        train_metrics = {}
        feature_importances = {}

        if labelled_pairs and any(p.label is not None for p in pairs):
            log.info("Labelled data available – training supervised model …")
            train_metrics       = ranker.train(pairs)
            feature_importances = ranker.feature_importances
            candidates          = ranker.rank(pairs)
        else:
            log.info("No labelled data – using unsupervised composite scoring …")
            candidates = ranker.rank_unsupervised(pairs)


        cluster_map = ranker.suggest_clusters(units, structural["tfidf_matrix"])

        top_candidates = candidates[:cfg.top_k]
        meta["stages"]["ranking"] = {
            "total_candidates": len(candidates),
            "top_k":            cfg.top_k,
            "mode":             "supervised" if train_metrics else "unsupervised",
            "elapsed_s":        round(time.time() - t_s, 2),
            "train_metrics":    train_metrics,
        }

        # ── Report ──────────────────────────────────────────────────
        total_elapsed = round(time.time() - t0, 2)
        meta["total_elapsed_s"] = total_elapsed
        log.info("Pipeline complete in %.1fs", total_elapsed)

        result = PipelineResult(
            units            = units,
            top_candidates   = top_candidates,
            cluster_map      = cluster_map,
            feature_importances = feature_importances,
            hotspots         = hotspots,
            meta             = meta,
        )
        self._save_report(result)
        return result


    @staticmethod
    def _load_labels(path: str) -> Dict:
        """Load ground truth labels from JSON file."""
        try:
            with open(path) as f:
                items = json.load(f)
            labels = {}
            for item in items:
                a, b   = item["comp_a"], item["comp_b"]
                key    = (min(a,b), max(a,b))
                labels[key] = int(item["label"])
            log.info("Loaded %d labelled pairs from %s", len(labels), path)
            return labels
        except Exception as e:
            log.warning("Could not load labels from %s: %s", path, e)
            return {}

    @staticmethod
    def _save_report(result: PipelineResult) -> None:
        report = {
            "meta":              result.meta,
            "top_boundaries":    [c.to_dict() for c in result.top_candidates],
            "service_clusters":  {str(k): v for k, v in result.cluster_map.items()},
            "feature_importances": result.feature_importances,
            "hotspots":          result.hotspots,
        }
        out_path = OUTPUT_DIR / "boundary_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Report saved to %s", out_path)
