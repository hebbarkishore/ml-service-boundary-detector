import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import git
    _HAS_GIT = True
except ImportError:
    _HAS_GIT = False
    logging.warning("gitpython not installed – evolutionary signals will be zero.")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GIT_MAX_COMMITS, GIT_SINCE_DAYS, CO_CHANGE_MIN_COUNT, SEQUENCE_LEAD_WINDOW_DAYS

log = logging.getLogger(__name__)

_DECAY_HALF_LIFE = 90.0



class EvolutionarySignalExtractor:
    """
    Walks the git history of a repository and computes co-change metrics
    for every pair of source files.  The results are then joined onto
    the CodeUnit list to produce pair-level evolutionary features.

    Usage
    -----
    extractor = EvolutionarySignalExtractor(repo_path="/path/to/repo")
    extractor.mine()
    features  = extractor.get_pair_features(units)
    """

    def __init__(self, repo_path: str):
        self._repo_path = repo_path
        self._repo: Optional["git.Repo"] = None

        self._file_change_count: Dict[str, int]       = defaultdict(int)
        self._co_change_raw:     Dict[Tuple, int]     = defaultdict(int)
        self._co_change_decay:   Dict[Tuple, float]   = defaultdict(float)
        # Ordered commit timestamps per file, used to compute sequence directionality.
        self._file_commit_times: Dict[str, List[float]] = defaultdict(list)
        self._mined = False


    def mine(self) -> None:
        """Mine the git repository. Must be called before get_pair_features."""
        if not _HAS_GIT:
            log.warning("gitpython unavailable - skipping git mining.")
            self._mined = True
            return

        try:
            self._repo = git.Repo(self._repo_path, search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            log.warning("No git repo found at %s - evolutionary signals zeroed.", self._repo_path)
            self._mined = True
            return

        since_date = datetime.utcnow() - timedelta(days=GIT_SINCE_DAYS)
        log.info("Mining git history since %s (max %d commits)…",
                 since_date.date(), GIT_MAX_COMMITS)

        commits_processed = 0
        now_ts = datetime.utcnow().timestamp()

        try:
            for commit in self._repo.iter_commits(
                since=since_date.strftime("%Y-%m-%d"),
                max_count=GIT_MAX_COMMITS,
            ):
                changed_files = self._changed_files(commit)
                if not changed_files:
                    continue

                age_days   = (now_ts - commit.committed_date) / 86400.0
                decay_w    = math.pow(0.5, age_days / _DECAY_HALF_LIFE)

                for fp in changed_files:
                    self._file_change_count[fp] += 1
                    self._file_commit_times[fp].append(float(commit.committed_date))

                if len(changed_files) <= 100:
                    files_list = sorted(changed_files)
                    for i, fa in enumerate(files_list):
                        for fb in files_list[i + 1:]:
                            key = (fa, fb)
                            self._co_change_raw[key]   += 1
                            self._co_change_decay[key] += decay_w

                commits_processed += 1

        except git.GitCommandError as e:
            log.warning("Git command error during mining: %s", e)

        log.info("Processed %d commits; %d unique co-change pairs found.",
                 commits_processed, len(self._co_change_raw))
        self._mined = True

    def get_pair_features(
        self,
        units,   # List[CodeUnit]
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Maps file-level co-change metrics onto unit-pairs.
        Returns dict[(unit_id_a, unit_id_b)] = {co_change_frequency, co_change_recency, logical_coupling_score}
        """
        if not self._mined:
            self.mine()

        fp_to_uid: Dict[str, str] = {}
        for u in units:
            fp_to_uid[self._normalise(u.file_path)] = u.unit_id

        unit_change_count: Dict[str, int]         = defaultdict(int)
        unit_co_raw:       Dict[Tuple, int]       = defaultdict(int)
        unit_co_decay:     Dict[Tuple, float]     = defaultdict(float)

        for fp, count in self._file_change_count.items():
            uid = fp_to_uid.get(self._normalise(fp))
            if uid:
                unit_change_count[uid] += count

        for (fa, fb), count in self._co_change_raw.items():
            if count < CO_CHANGE_MIN_COUNT:
                continue
            ua = fp_to_uid.get(self._normalise(fa))
            ub = fp_to_uid.get(self._normalise(fb))
            if ua and ub and ua != ub:
                key = (min(ua, ub), max(ua, ub))
                unit_co_raw[key]   += count
                unit_co_decay[key] += self._co_change_decay.get((fa, fb), 0.0)

        # Build per-unit commit-time lists for sequence directionality.
        unit_commit_times: Dict[str, List[float]] = defaultdict(list)
        for fp, times in self._file_commit_times.items():
            uid = fp_to_uid.get(self._normalise(fp))
            if uid:
                unit_commit_times[uid].extend(times)
        for uid in unit_commit_times:
            unit_commit_times[uid].sort()

        result: Dict[Tuple[str, str], Dict[str, float]] = {}
        all_change_total = max(sum(unit_change_count.values()), 1)

        for (ua, ub), raw in unit_co_raw.items():
            ca = unit_change_count.get(ua, 1)
            cb = unit_change_count.get(ub, 1)
            denom = math.sqrt(ca * cb)
            logical_coupling = raw / denom if denom > 0 else 0.0

            directionality = self._sequence_directionality(
                unit_commit_times.get(ua, []),
                unit_commit_times.get(ub, []),
            )

            result[(ua, ub)] = {
                "co_change_frequency":         raw / all_change_total,
                "co_change_recency":           unit_co_decay.get((ua, ub), 0.0),
                "logical_coupling_score":      logical_coupling,
                "change_sequence_directionality": directionality,
            }
        return result

    def get_change_hotspots(self, units, top_k: int = 10) -> List[Dict]:
        """Return most frequently changed units (churn hotspots)."""
        from collections import Counter
        fp_to_uid = {self._normalise(u.file_path): u.unit_id for u in units}
        uid_counts: Counter = Counter()
        for fp, cnt in self._file_change_count.items():
            uid = fp_to_uid.get(self._normalise(fp))
            if uid:
                uid_counts[uid] += cnt
        return [{"unit_id": uid, "change_count": cnt}
                for uid, cnt in uid_counts.most_common(top_k)]

    def get_commit_metadata(self, max_commits: int = 20) -> List[Dict]:
        """Return recent commit summaries for report context."""
        if not self._repo:
            return []
        commits = []
        for c in self._repo.iter_commits(max_count=max_commits):
            commits.append({
                "sha":     c.hexsha[:8],
                "author":  str(c.author),
                "date":    datetime.fromtimestamp(c.committed_date).isoformat(),
                "message": c.message.strip().splitlines()[0][:100],
            })
        return commits


    @staticmethod
    def _sequence_directionality(
        times_a: List[float],
        times_b: List[float],
    ) -> float:
        """
        Compute how consistently A's commits precede B's commits (or vice versa)
        within a SEQUENCE_LEAD_WINDOW_DAYS window.

        For each change to A, count whether B changed within the next
        SEQUENCE_LEAD_WINDOW_DAYS days (lead_ab).  Do the same with A and B
        swapped (lead_ba).  Return the normalised asymmetry in [0, 1]:
          0.0 = no directional pattern / perfectly symmetric
          1.0 = one file always leads the other
        """
        if not times_a or not times_b:
            return 0.0

        window_secs = SEQUENCE_LEAD_WINDOW_DAYS * 86400.0
        lead_ab = 0  # A changed, then B changed within window
        lead_ba = 0  # B changed, then A changed within window

        b_arr = times_b  # already sorted

        import bisect
        for ta in times_a:
            lo = bisect.bisect_right(b_arr, ta)
            hi = bisect.bisect_right(b_arr, ta + window_secs)
            if hi > lo:
                lead_ab += 1

        a_arr = times_a
        for tb in times_b:
            lo = bisect.bisect_right(a_arr, tb)
            hi = bisect.bisect_right(a_arr, tb + window_secs)
            if hi > lo:
                lead_ba += 1

        total = lead_ab + lead_ba
        return abs(lead_ab - lead_ba) / total if total > 0 else 0.0

    @staticmethod
    def _changed_files(commit: "git.Commit") -> Set[str]:
        """Get set of changed file paths in a commit."""
        try:
            if commit.parents:
                diff = commit.parents[0].diff(commit)
                return {
                    item.b_path or item.a_path
                    for item in diff
                    if item.b_path or item.a_path
                }
            else:
                return {item.path for item in commit.tree.traverse()
                        if hasattr(item, "path")}
        except Exception as e:
            log.debug("Could not diff commit %s: %s", commit.hexsha[:8], e)
            return set()

    @staticmethod
    def _normalise(path: str) -> str:
        """Normalise path separators and strip leading ./ for map lookups."""
        return Path(path).as_posix().lstrip("./")
