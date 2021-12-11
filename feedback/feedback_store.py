import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_DIR

log = logging.getLogger(__name__)

FEEDBACK_FILE = OUTPUT_DIR / "feedback_store.json"



@dataclass
class FeedbackEntry:
    """
    A single architect accept/reject decision on a boundary candidate pair.
    label=1  → ACCEPT  (this IS a valid service boundary)
    label=0  → REJECT  (these components should stay together)
    """
    comp_a:         str
    comp_b:         str
    label:          int            
    boundary_score: float = 0.0   
    confidence:     float = 0.0
    rationale_text: str = ""      
    decided_by:     str = "architect"
    decided_at:     str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    run_id:         str = ""      

    def canonical_key(self) -> str:
        """Canonical key regardless of (A,B) vs (B,A) order."""
        a, b = sorted([self.comp_a, self.comp_b])
        return f"{a}|||{b}"



class FeedbackStore:
    """
    Reads and writes architect decisions to a persistent JSON store.
    Thread-safe for single-process use (Flask dev server).
    """

    def __init__(self, store_path: Optional[str] = None):
        self._path = Path(store_path) if store_path else FEEDBACK_FILE
        self._entries: Dict[str, FeedbackEntry] = {}
        self._load()


    def record(
        self,
        comp_a:         str,
        comp_b:         str,
        label:          int,
        boundary_score: float = 0.0,
        confidence:     float = 0.0,
        rationale_text: str   = "",
        decided_by:     str   = "architect",
        run_id:         str   = "",
    ) -> FeedbackEntry:
        """
        Record a new accept/reject decision.
        If a decision already exists for this pair, it is OVERWRITTEN
        (architect can change their mind).
        """
        entry = FeedbackEntry(
            comp_a         = comp_a,
            comp_b         = comp_b,
            label          = int(label),
            boundary_score = boundary_score,
            confidence     = confidence,
            rationale_text = rationale_text,
            decided_by     = decided_by,
            decided_at     = datetime.utcnow().isoformat(),
            run_id         = run_id,
        )
        key = entry.canonical_key()
        prev = self._entries.get(key)
        if prev:
            log.info(
                "Overwriting previous decision for (%s, %s): %d → %d",
                comp_a, comp_b, prev.label, label
            )
        self._entries[key] = entry
        self._save()
        log.info(
            "Feedback recorded: (%s) ↔ (%s) = %s",
            self._short(comp_a), self._short(comp_b),
            "ACCEPT (boundary)" if label == 1 else "REJECT (keep together)"
        )
        return entry

    def bulk_record(self, decisions: List[Dict]) -> int:
        """
        Record multiple decisions at once.
        decisions: list of dicts with keys comp_a, comp_b, label (required)
                   and optionally boundary_score, confidence, rationale_text
        Returns count of successfully recorded entries.
        """
        count = 0
        for d in decisions:
            try:
                self.record(
                    comp_a         = d["comp_a"],
                    comp_b         = d["comp_b"],
                    label          = int(d["label"]),
                    boundary_score = float(d.get("boundary_score", 0.0)),
                    confidence     = float(d.get("confidence", 0.0)),
                    rationale_text = d.get("rationale_text", ""),
                    decided_by     = d.get("decided_by", "architect"),
                    run_id         = d.get("run_id", ""),
                )
                count += 1
            except (KeyError, ValueError) as e:
                log.warning("Skipping invalid feedback entry %s: %s", d, e)
        return count

    def delete(self, comp_a: str, comp_b: str) -> bool:
        """Remove a specific decision (architect changed mind, wants fresh look)."""
        key = FeedbackEntry(comp_a=comp_a, comp_b=comp_b, label=0).canonical_key()
        if key in self._entries:
            del self._entries[key]
            self._save()
            return True
        return False

    def clear_all(self) -> int:
        """Wipe the entire store. Returns count of deleted entries."""
        count = len(self._entries)
        self._entries.clear()
        self._save()
        log.warning("Feedback store cleared (%d entries deleted)", count)
        return count



    def get_all_labels(self) -> Dict[Tuple[str, str], int]:
        """
        Returns {(comp_a, comp_b): label} for use by the ML pipeline.
        Keys are always (min, max) sorted for consistent lookup.
        """
        result = {}
        for entry in self._entries.values():
            key = (min(entry.comp_a, entry.comp_b),
                   max(entry.comp_a, entry.comp_b))
            result[key] = entry.label
        return result

    def get_labelled_pairs_for_training(self) -> List[Dict]:
        """
        Returns list of dicts compatible with the ML pipeline's label format.
        """
        return [
            {
                "comp_a": e.comp_a,
                "comp_b": e.comp_b,
                "label":  e.label,
            }
            for e in self._entries.values()
        ]

    def get_all_entries(self) -> List[FeedbackEntry]:
        return list(self._entries.values())

    def get_entry(self, comp_a: str, comp_b: str) -> Optional[FeedbackEntry]:
        key = FeedbackEntry(comp_a=comp_a, comp_b=comp_b, label=0).canonical_key()
        return self._entries.get(key)

    def stats(self) -> Dict:
        entries = list(self._entries.values())
        accepted = [e for e in entries if e.label == 1]
        rejected = [e for e in entries if e.label == 0]
        return {
            "total":    len(entries),
            "accepted": len(accepted),   
            "rejected": len(rejected),   
            "last_updated": max(
                (e.decided_at for e in entries), default="never"
            ),
        }

    def is_sufficient_for_training(self, min_samples: int = 10) -> bool:
        """Returns True if we have enough labelled data to train supervised model."""
        stats = self.stats()
        # Need both positive and negative examples
        return stats["accepted"] >= 3 and stats["rejected"] >= 3 and stats["total"] >= min_samples


    def _load(self) -> None:
        if not self._path.exists():
            log.debug("No feedback store found at %s — starting fresh.", self._path)
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw.get("entries", []):
                entry = FeedbackEntry(**item)
                self._entries[entry.canonical_key()] = entry
            log.info(
                "Loaded %d feedback entries from %s", len(self._entries), self._path
            )
        except Exception as e:
            log.error("Failed to load feedback store: %s", e)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version":  "1.0",
                "updated":  datetime.utcnow().isoformat(),
                "stats":    self.stats(),
                "entries":  [asdict(e) for e in self._entries.values()],
            }
            
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self._path)
        except Exception as e:
            log.error("Failed to save feedback store: %s", e)

    @staticmethod
    def _short(uid: str, n: int = 30) -> str:
        return uid if len(uid) <= n else "…" + uid[-(n-1):]
