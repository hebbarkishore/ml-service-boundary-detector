import re
import os
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EXECUTION_ORDER_MIN_CALLS

log = logging.getLogger(__name__)




_PATTERNS = {

    "log4j": re.compile(
        r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})"   
        r".*?\[([A-Za-z][\w$.]+)\]"                       
    ),


    "python_log": re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"[,\.]?\d*\s+-\s+([\w.]+)\s+-\s+\w+"
    ),


    "call_notation": re.compile(
        r"CALL\s+([\w$.]+)\s*->\s*([\w$.]+)"
        r"(?:\s*\[depth=(\d+)\])?"
    ),


    "otel_csv": re.compile(
        r"^[0-9a-f\-]{8,},\s*[0-9a-f\-]{8,},\s*([0-9a-f\-]*),\s*"
        r"([\w.\-]+),\s*([\w./\-]+),\s*\d+,\s*\d+"
    ),


    "spring_http": re.compile(
        r"(GET|POST|PUT|DELETE|PATCH)\s+(/\S+)\s+(\d{3})\s+\d+ms\s+([\w.]+)"
    ),
}


_WINDOW_SECONDS = 5




class _TraceEvent:
    __slots__ = ("timestamp", "component", "target", "depth")

    def __init__(self, timestamp: float, component: str,
                 target: Optional[str] = None, depth: int = 0):
        self.timestamp = timestamp
        self.component = component
        self.target    = target
        self.depth     = depth




class BehavioralSignalExtractor:
    """
    Reads one or more trace / log files and builds a behavioral feature
    matrix for component pairs.

    Usage
    -----
    extractor = BehavioralSignalExtractor(unit_ids)
    extractor.load_trace_files(["/var/log/app.log", "traces/otel_export.csv"])
    features  = extractor.get_pair_features()
    """

    def __init__(self, unit_ids: List[str]):
        self._unit_ids  = unit_ids
        self._unit_set  = set(unit_ids)
        self._unit_index = {u: i for i, u in enumerate(unit_ids)}
        n = len(unit_ids)

        # Accumulation matrices
        self._call_count    = np.zeros((n, n), dtype=np.float32)  # direct calls
        self._call_depth    = np.zeros((n, n), dtype=np.float32)  # sum of depths
        self._depth_count   = np.zeros((n, n), dtype=np.float32)  # for averaging
        self._co_occurrence = np.zeros((n, n), dtype=np.float32)  # window co-occurrence
        # Execution order stability
        self._order_before  = np.zeros((n, n), dtype=np.float32)

        self._total_windows   = 0
        self._events_parsed   = 0



    def load_trace_files(self, file_paths: List[str]) -> None:
        """Load and parse all provided trace / log files."""
        for fp in file_paths:
            if not Path(fp).is_file():
                log.warning("Trace file not found: %s", fp)
                continue
            log.info("Parsing trace file: %s", fp)
            events = self._parse_file(fp)
            self._accumulate(events)
            log.info("  → %d trace events extracted", len(events))

    def get_pair_features(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Returns dict[(comp_a, comp_b)] = {
            runtime_call_frequency, runtime_call_depth, temporal_affinity,
            execution_order_stability
        }
        Pairs are keyed in canonical (min, max) order.
        """
        result = {}
        n = len(self._unit_ids)
        total_calls   = self._call_count.sum()
        total_windows = max(self._total_windows, 1)

        for i in range(n):
            for j in range(i + 1, n):
                c_ij = self._call_count[i, j]
                c_ji = self._call_count[j, i]
                co   = self._co_occurrence[i, j]
                if c_ij == 0 and c_ji == 0 and co == 0:
                    continue

                c_dom = max(c_ij, c_ji)
                avg_depth = (self._call_depth[i, j] / self._depth_count[i, j]
                             if self._depth_count[i, j] > 0 else 0.0)
                affinity  = co / total_windows

                # Execution order stability: how consistently one component
                # precedes the other across co-occurrence windows.
                # Range [0, 1]; 1.0 = perfectly unidirectional order.
                o_ij = self._order_before[i, j]
                o_ji = self._order_before[j, i]
                order_total = o_ij + o_ji
                if order_total >= EXECUTION_ORDER_MIN_CALLS:
                    stability = abs(o_ij - o_ji) / order_total
                else:
                    # Fall back to call-direction asymmetry when timestamped
                    # events are too few to be reliable.
                    call_total = c_ij + c_ji
                    stability  = (abs(c_ij - c_ji) / call_total
                                  if call_total >= EXECUTION_ORDER_MIN_CALLS else 0.0)

                key = (self._unit_ids[i], self._unit_ids[j])
                result[key] = {
                    "runtime_call_frequency":    float(c_dom / max(total_calls, 1)),
                    "runtime_call_depth":        float(avg_depth),
                    "temporal_affinity":         float(affinity),
                    "execution_order_stability": float(stability),
                }
        return result

    def get_hotpaths(self, top_k: int = 10) -> List[Dict]:
        """Returns the top-k most frequent direct call pairs."""
        pairs = []
        n = len(self._unit_ids)
        for i in range(n):
            for j in range(n):
                if self._call_count[i, j] > 0:
                    pairs.append({
                        "from":  self._unit_ids[i],
                        "to":    self._unit_ids[j],
                        "calls": int(self._call_count[i, j]),
                    })
        return sorted(pairs, key=lambda x: x["calls"], reverse=True)[:top_k]



    def _parse_file(self, fp: str) -> List[_TraceEvent]:
        lines  = Path(fp).read_text(encoding="utf-8", errors="ignore").splitlines()
        events: List[_TraceEvent] = []

        for line in lines:
            m = _PATTERNS["call_notation"].search(line)
            if m:
                src, tgt = m.group(1), m.group(2)
                depth     = int(m.group(3)) if m.group(3) else 0
                src_id    = self._resolve(src)
                tgt_id    = self._resolve(tgt)
                if src_id and tgt_id:
                    events.append(_TraceEvent(0.0, src_id, tgt_id, depth))
                continue

            m = _PATTERNS["otel_csv"].match(line)
            if m:
                svc = self._resolve(m.group(2))
                if svc:
                    events.append(_TraceEvent(0.0, svc))
                continue

            m = _PATTERNS["log4j"].search(line)
            if m:
                ts   = self._parse_ts(m.group(1))
                comp = self._resolve(m.group(2))
                if comp:
                    events.append(_TraceEvent(ts, comp))
                continue

            m = _PATTERNS["python_log"].search(line)
            if m:
                ts   = self._parse_ts(m.group(1))
                comp = self._resolve(m.group(2))
                if comp:
                    events.append(_TraceEvent(ts, comp))
                continue

        return events

    def _resolve(self, name: str) -> Optional[str]:
        """Fuzzy-match a log token to a known unit_id."""
        if name in self._unit_set:
            return name
        for uid in self._unit_ids:
            if uid.endswith("." + name) or uid.endswith("/" + name) or uid == name:
                return uid
            short = uid.split(".")[-1]
            if short == name or short == name.split(".")[-1]:
                return uid
        return None


    def _accumulate(self, events: List[_TraceEvent]) -> None:
        self._events_parsed += len(events)

        for ev in events:
            if ev.target:
                i = self._unit_index.get(ev.component)
                j = self._unit_index.get(ev.target)
                if i is not None and j is not None:
                    self._call_count[i, j]  += 1
                    self._call_depth[i, j]  += ev.depth
                    self._depth_count[i, j] += 1

        ts_events = [ev for ev in events if ev.timestamp > 0 and not ev.target]
        if not ts_events:
            return

        ts_events.sort(key=lambda e: e.timestamp)
        window: List[_TraceEvent] = []

        for ev in ts_events:
            window = [e for e in window
                      if ev.timestamp - e.timestamp <= _WINDOW_SECONDS]
            i = self._unit_index.get(ev.component)
            seen_in_window = {e.component for e in window}
            for other in seen_in_window:
                j = self._unit_index.get(other)
                if i is not None and j is not None and i != j:
                    self._co_occurrence[i, j] += 1
                    self._co_occurrence[j, i] += 1
                    self._order_before[j, i] += 1
            window.append(ev)
            self._total_windows += 1

    @staticmethod
    def _parse_ts(ts_str: str) -> float:
        """Parse ISO-ish timestamp string to epoch float (best-effort)."""
        import time, datetime
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.datetime.strptime(ts_str.strip(), fmt).timestamp()
            except ValueError:
                continue
        return 0.0
