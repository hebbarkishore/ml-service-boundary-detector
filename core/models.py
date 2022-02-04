"""
core/models.py  –  Shared data-transfer objects used across all pipeline stages.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CodeUnit:
    """
    Represents one logical code unit – typically a class (Java) or module (Python).
    This is the primary node in every signal graph.
    """
    unit_id:      str          
    file_path:    str
    language:     str
    package:      str = ""

    class_names:  List[str] = field(default_factory=list)
    method_names: List[str] = field(default_factory=list)
    imports:      List[str] = field(default_factory=list)
    annotations:  List[str] = field(default_factory=list)   
    comments:     str = ""                                  
    raw_tokens:   List[str] = field(default_factory=list)   


    domain_hints: List[str] = field(default_factory=list)   


    cyclomatic_complexity: float = 0.0
    loc: int = 0

    def vocabulary(self) -> str:
        """Combined textual vocabulary for TF-IDF / semantic embedding."""
        return " ".join(
            self.class_names
            + self.method_names
            + self.annotations
            + self.raw_tokens
            + [self.comments]
        )


@dataclass
class DependencyEdge:
    source: str         
    target: str          
    kind:   str          
    weight: int = 1



@dataclass
class DocumentChunk:
    """
    A chunk of text extracted from a requirements / architecture document.
    Used to enrich the vocabulary for semantic similarity.
    """
    source_file: str
    text:        str
    doc_type:    str = "unknown"   # "requirements" | "architecture" | "api-spec" | …



@dataclass
class PairFeatures:
    """
    All signals for a (component_a, component_b) pair that the ML model consumes.
    """
    comp_a: str
    comp_b: str

    # Structural
    structural_coupling_weight: float = 0.0    
    tfidf_cosine_similarity:    float = 0.0    
    semantic_similarity:        float = 0.0    
    shared_import_count:        int   = 0      
    shared_annotation_count:    int   = 0      
    inheritance_linked:         int   = 0      

    # Behavioral
    runtime_call_frequency:     float = 0.0
    runtime_call_depth:         float = 0.0
    temporal_affinity:          float = 0.0
    execution_order_stability:  float = 0.0

    # Evolutionary
    co_change_frequency:        float = 0.0    
    co_change_recency:          float = 0.0    
    logical_coupling_score:     float = 0.0    

    # Graph centrality features
    pagerank_a:     float = 0.0
    pagerank_b:     float = 0.0
    betweenness_a:  float = 0.0
    betweenness_b:  float = 0.0

    # Cross-layer indicator 
    cross_layer_flag: int = 0

    # Ground truth label (1 = valid boundary, 0 = should stay together)
    label: Optional[int] = None

    def to_feature_vector(self) -> List[float]:
        """Returns ordered feature list for sklearn estimators."""
        return [
            self.structural_coupling_weight,
            self.tfidf_cosine_similarity,
            self.semantic_similarity,
            float(self.shared_import_count),
            float(self.shared_annotation_count),
            float(self.inheritance_linked),
            self.runtime_call_frequency,
            self.runtime_call_depth,
            self.temporal_affinity,
            self.execution_order_stability,
            self.co_change_frequency,
            self.co_change_recency,
            self.logical_coupling_score,
            self.pagerank_a,
            self.pagerank_b,
            self.betweenness_a,
            self.betweenness_b,
            float(self.cross_layer_flag),
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "structural_coupling_weight",
            "tfidf_cosine_similarity",
            "semantic_similarity",
            "shared_import_count",
            "shared_annotation_count",
            "inheritance_linked",
            "runtime_call_frequency",
            "runtime_call_depth",
            "temporal_affinity",
            "execution_order_stability",
            "co_change_frequency",
            "co_change_recency",
            "logical_coupling_score",
            "pagerank_a",
            "pagerank_b",
            "betweenness_a",
            "betweenness_b",
            "cross_layer_flag",
        ]

@dataclass
class BoundaryCandidate:
    comp_a:          str
    comp_b:          str
    boundary_score:  float          
    confidence:      float         
    rationale:       Dict[str, float] = field(default_factory=dict)
    suggested_service: Optional[str] = None   # proposed service name

    def to_dict(self) -> dict:
        return {
            "component_a":       self.comp_a,
            "component_b":       self.comp_b,
            "boundary_score":    round(self.boundary_score, 4),
            "confidence":        round(self.confidence, 4),
            "suggested_service": self.suggested_service,
            "rationale":         {k: round(v, 4) for k, v in self.rationale.items()},
        }
