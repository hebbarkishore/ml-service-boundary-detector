import ast
import os
import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import javalang
    _HAS_JAVA = True
except ImportError:
    _HAS_JAVA = False
    logging.warning("javalang not installed - Java files will be token-only.")

try:
    import lizard
    _HAS_LIZARD = True
except ImportError:
    _HAS_LIZARD = False

try:
    from gensim.models import Word2Vec
    _HAS_GENSIM = True
except ImportError:
    _HAS_GENSIM = False
    logging.warning("gensim not installed - semantic similarity will be skipped.")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.models import CodeUnit, DependencyEdge
from config import (CODE_EXTENSIONS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
                    TFIDF_MIN_DF, COUPLING_STRONG_THR)

log = logging.getLogger(__name__)



_CAMEL_RE = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?=[A-Z]|$)")
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              "build", "dist", "target", ".gradle", ".idea", ".mvn"}

LAYER_PATTERNS = {
    "service":    re.compile(r"service|usecase|use_case|business", re.I),
    "repository": re.compile(r"repo|repository|dao|store|persist", re.I),
    "controller": re.compile(r"controller|handler|view|endpoint|route|resource", re.I),
    "model":      re.compile(r"model|entity|domain|schema|dto|vo\b", re.I),
    "util":       re.compile(r"util|helper|common|shared|mixin|support", re.I),
}

SPRING_ANNOTATIONS = {
    "RestController", "Controller", "Service", "Repository", "Component",
    "Configuration", "Bean", "Entity", "FeignClient", "KafkaListener",
}


def _tokenise(text: str) -> List[str]:
    """Split camelCase / snake_case identifiers into lowercase tokens."""
    tokens = _CAMEL_RE.findall(text)
    tokens += re.findall(r"[a-z]{3,}", text)
    return [t.lower() for t in tokens if len(t) > 2]


def _layer_hints(name: str) -> List[str]:
    return [layer for layer, pat in LAYER_PATTERNS.items() if pat.search(name)]



class _PythonParser:

    def parse(self, path: str) -> Optional[CodeUnit]:
        src = Path(path).read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError as e:
            log.warning("SyntaxError %s: %s", path, e)
            return self._fallback(path, src)

        unit = CodeUnit(
            unit_id=self._module_id(path),
            file_path=path,
            language="python",
        )
        comments = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                unit.class_names.append(node.name)
                unit.domain_hints.extend(_layer_hints(node.name))
                ds = ast.get_docstring(node)
                if ds:
                    comments.append(ds)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                unit.method_names.append(node.name)
                ds = ast.get_docstring(node)
                if ds:
                    comments.append(ds)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    unit.imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    unit.imports.append(node.module)

        for line in src.splitlines():
            s = line.strip()
            if s.startswith("#"):
                comments.append(s.lstrip("# "))

        unit.comments   = " ".join(comments)
        unit.raw_tokens = _tokenise(src)
        unit            = self._lizard_metrics(unit, path)
        return unit

    def _fallback(self, path: str, src: str) -> CodeUnit:
        return CodeUnit(unit_id=Path(path).stem, file_path=path,
                        language="python", raw_tokens=_tokenise(src))

    @staticmethod
    def _module_id(path: str) -> str:
        parts = list(Path(path).with_suffix("").parts)
        skip  = {"src", "app", "lib", "pkg", "main", "python"}
        while parts and parts[0].lower() in skip:
            parts.pop(0)
        return ".".join(parts) if parts else Path(path).stem

    @staticmethod
    def _lizard_metrics(unit: CodeUnit, path: str) -> CodeUnit:
        if not _HAS_LIZARD:
            return unit
        try:
            analysis = lizard.analyze_file(path)
            if analysis.function_list:
                unit.cyclomatic_complexity = float(
                    np.mean([f.cyclomatic_complexity for f in analysis.function_list]))
            unit.loc = analysis.nloc
        except Exception as e:
            log.debug("lizard error %s: %s", path, e)
        return unit



class _JavaParser:

    def parse(self, path: str) -> Optional[CodeUnit]:
        src = Path(path).read_text(encoding="utf-8", errors="ignore")
        if not _HAS_JAVA:
            return self._token_only(path, src)
        try:
            tree = javalang.parse.parse(src)
        except Exception as e:
            log.warning("Java parse error %s: %s", path, e)
            return self._token_only(path, src)

        pkg  = tree.package.name if tree.package else ""
        unit = CodeUnit(unit_id="", file_path=path, language="java", package=pkg)
        comments: List[str] = []

        for _, cls in tree.filter(javalang.tree.ClassDeclaration):
            fqn = f"{pkg}.{cls.name}" if pkg else cls.name
            unit.class_names.append(fqn)
            if not unit.unit_id:
                unit.unit_id = fqn
            unit.domain_hints.extend(_layer_hints(cls.name))
            for ann in (cls.annotations or []):
                unit.annotations.append(ann.name)
                if ann.name in SPRING_ANNOTATIONS:
                    unit.domain_hints.append(ann.name.lower())
            for method in (cls.methods or []):
                unit.method_names.append(method.name)

        for imp in (tree.imports or []):
            unit.imports.append(imp.path)

        comments += re.findall(r"/\*\*?(.*?)\*/", src, re.DOTALL)
        comments += re.findall(r"//(.+)", src)
        unit.comments   = " ".join(comments)
        unit.raw_tokens = _tokenise(src)
        if not unit.unit_id:
            unit.unit_id = Path(path).stem
        unit = _PythonParser._lizard_metrics(unit, path)
        return unit

    @staticmethod
    def _token_only(path: str, src: str) -> CodeUnit:
        return CodeUnit(unit_id=Path(path).stem, file_path=path,
                        language="java", raw_tokens=_tokenise(src))



class _GenericParser:
    def parse(self, path: str) -> CodeUnit:
        src = Path(path).read_text(encoding="utf-8", errors="ignore")
        return CodeUnit(
            unit_id    = Path(path).stem,
            file_path  = path,
            language   = CODE_EXTENSIONS.get(Path(path).suffix.lower(), "unknown"),
            raw_tokens = _tokenise(src),
            comments   = " ".join(re.findall(r"//(.+)|#(.+)", src)),
        )



class StructuralSignalExtractor:

    def __init__(self):
        self._py  = _PythonParser()
        self._jv  = _JavaParser()
        self._gen = _GenericParser()


    def extract(
        self,
        roots: List[str],                   
        doc_corpus: Optional[List[str]] = None,   
    ) -> Dict:
        """
        Returns a dict with:
          units           - List[CodeUnit]
          edges           - List[DependencyEdge]
          graph           - nx.DiGraph
          tfidf_matrix    - sparse CSR (n_units × n_features)
          tfidf_vocab     - vectoriser vocabulary
          cosine_sim      - n×n cosine similarity matrix
          semantic_sim    - n×n semantic similarity matrix (or zeros)
          centrality      - dict {unit_id: {pagerank, betweenness}}
          unit_index      - {unit_id: row_index}
        """
        files = self._collect(roots)
        units = self._parse(files)
        log.info("Parsed %d code units", len(units))

        edges = self._build_edges(units)
        graph = self._build_graph(units, edges)

        corpus    = [u.vocabulary() for u in units]
        unit_index = {u.unit_id: i for i, u in enumerate(units)}

       
        tfidf_vec    = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=max(1, TFIDF_MIN_DF) if len(units) > 5 else 1,
        )
        tfidf_matrix = tfidf_vec.fit_transform(corpus)
        cosine_sim   = cosine_similarity(tfidf_matrix).astype(np.float32)

        semantic_sim = self._word2vec_similarity(units, doc_corpus)


        centrality = self._centrality(graph)

        return dict(
            units       = units,
            edges       = edges,
            graph       = graph,
            tfidf_matrix= tfidf_matrix,
            tfidf_vocab = tfidf_vec.vocabulary_,
            cosine_sim  = cosine_sim,
            semantic_sim= semantic_sim,
            centrality  = centrality,
            unit_index  = unit_index,
        )


    def _collect(self, roots: List[str]) -> List[str]:
        files = []
        for root in roots:
            p = Path(root)
            if p.is_file():
                if p.suffix.lower() in CODE_EXTENSIONS:
                    files.append(str(p))
            elif p.is_dir():
                for dp, dirs, fnames in os.walk(str(p)):
                    dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
                    for fn in fnames:
                        if Path(fn).suffix.lower() in CODE_EXTENSIONS:
                            files.append(os.path.join(dp, fn))
        log.info("Collected %d source files", len(files))
        return files

    def _parse(self, files: List[str]) -> List[CodeUnit]:
        units = []
        seen_ids: Dict[str, int] = {}
        for fp in files:
            ext  = Path(fp).suffix.lower()
            unit = (self._py.parse(fp)  if ext == ".py"   else
                    self._jv.parse(fp)  if ext == ".java"  else
                    self._gen.parse(fp))
            if unit is None:
                continue
            if unit.unit_id in seen_ids:
                seen_ids[unit.unit_id] += 1
                unit.unit_id = f"{unit.unit_id}_{seen_ids[unit.unit_id]}"
            else:
                seen_ids[unit.unit_id] = 0
            units.append(unit)
        return units


    def _build_edges(self, units: List[CodeUnit]) -> List[DependencyEdge]:
        id_set = {u.unit_id for u in units}
        prefix_map: Dict[str, str] = {}
        for u in units:
            for cls in u.class_names:
                prefix_map[cls]           = u.unit_id
                prefix_map[u.unit_id]     = u.unit_id
                short = cls.split(".")[-1]
                if short not in prefix_map:
                    prefix_map[short] = u.unit_id

        weight_map: Dict[Tuple[str,str], int] = defaultdict(int)
        for u in units:
            for imp in u.imports:
                target = self._resolve(imp, prefix_map)
                if target and target != u.unit_id:
                    weight_map[(u.unit_id, target)] += 1

        edges = []
        for (src, tgt), w in weight_map.items():
            kind = "strong_import" if w >= COUPLING_STRONG_THR else "import"
            edges.append(DependencyEdge(src, tgt, kind, w))
        return edges

    @staticmethod
    def _resolve(imp: str, prefix_map: Dict[str, str]) -> Optional[str]:
        """Best-effort import - unit_id resolution."""
        if imp in prefix_map:
            return prefix_map[imp]
        parts = imp.split(".")
        for i in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in prefix_map:
                return prefix_map[candidate]
            short = parts[i]
            if short in prefix_map:
                return prefix_map[short]
        return None


    @staticmethod
    def _build_graph(units: List[CodeUnit], edges: List[DependencyEdge]) -> nx.DiGraph:
        G = nx.DiGraph()
        for u in units:
            G.add_node(u.unit_id, language=u.language,
                       layer=u.domain_hints[0] if u.domain_hints else "unknown",
                       loc=u.loc, cc=u.cyclomatic_complexity)
        for e in edges:
            if G.has_edge(e.source, e.target):
                G[e.source][e.target]["weight"] += e.weight
            else:
                G.add_edge(e.source, e.target, weight=e.weight, kind=e.kind)
        return G


    @staticmethod
    def _centrality(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        if len(G) == 0:
            return {}
        pr = nx.pagerank(G, alpha=0.85, weight="weight")
        k = min(len(G), 50)
        try:
            bc = nx.betweenness_centrality(G, k=k, weight="weight", normalized=True)
        except Exception:
            bc = {n: 0.0 for n in G.nodes()}
        result = {}
        for node in G.nodes():
            result[node] = {"pagerank": pr.get(node, 0.0),
                            "betweenness": bc.get(node, 0.0)}
        return result


    @staticmethod
    def _word2vec_similarity(
        units: List[CodeUnit],
        doc_corpus: Optional[List[str]],
    ) -> np.ndarray:
        n = len(units)
        if not _HAS_GENSIM or n == 0:
            return np.zeros((n, n), dtype=np.float32)

        sentences = [u.raw_tokens for u in units if u.raw_tokens]
        if doc_corpus:
            for doc in doc_corpus:
                sentences.append(_tokenise(doc))

        if len(sentences) < 2:
            return np.zeros((n, n), dtype=np.float32)

        model = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            epochs=10,
            seed=42,
        )

        def unit_vector(tokens):
            vecs = [model.wv[t] for t in tokens if t in model.wv]
            if not vecs:
                return np.zeros(100)
            return np.mean(vecs, axis=0)

        vectors = np.array([unit_vector(u.raw_tokens) for u in units])
        norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        vectors /= norms

        sim = (vectors @ vectors.T).astype(np.float32)
        np.fill_diagonal(sim, 0.0)
        return sim
