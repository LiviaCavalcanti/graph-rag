"""CVE-aware CWE sampling for CVEfixes pair collections.

Extracted from ``cvefixes_experiments/scripts/performance/
exp_cvefixes_retrieval_grid_file_level.py`` so it can be reused by
:class:`src.data.cvefixes_file.CVEFixesFileDataset` (and any other caller)
instead of being duplicated.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

SAMPLE_MODES = ("original", "proportional", "balanced")


def sample_cve_aware(
    pairs: list,
    *,
    mode: str,
    seed: int = 42,
    min_cves_per_cwe: int = 1,
    total: int | None = None,
) -> tuple[list, dict[str, Any]]:
    """Sample pairs at CVE granularity to reshape (or preserve) the CWE distribution.

    Modes:
      - ``"original"``: no resampling — returns ``pairs`` unchanged, in order.
      - ``"proportional"``: keep the dataset's natural CWE proportions
        (CVE-grouped; optionally resampled down to ``total``).
      - ``"balanced"``: cap every CWE to the same budget so the distribution is
        approximately uniform across CWEs (under-samples the majority CWEs).

    CVE-aware guarantees (``proportional``/``balanced`` modes):
      - Whole CVE groups are kept together (a CVE never straddles the sample
        boundary), so a downstream leakage-safe split can always find
        same-CVE support for its queries.
      - At least ``min_cves_per_cwe`` CVEs are retained per CWE when available.
      - Multi-pair CVEs are preferred, so each CWE keeps CVEs that can serve as
        both index and query.

    Args:
        pairs: ``FunctionPair`` objects (graphs not required for sampling).
        mode: one of :data:`SAMPLE_MODES`.
        seed: RNG seed for reproducible CVE selection.
        min_cves_per_cwe: minimum CVEs kept per CWE (whole groups).
        total: target number of pairs. If ``None``: proportional keeps all
            pairs; balanced under-samples every CWE to the smallest CWE's
            pair count. Ignored for ``"original"``.

    Returns:
        ``(sampled_pairs, info)`` where ``info`` records the realized per-CWE
        CVE/pair counts and the requested target.
    """
    if mode == "original" or mode not in SAMPLE_MODES:
        return pairs[:], {"mode": mode or "original", "applied": False}

    rng = random.Random(seed)

    by_cwe: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        cwe = getattr(pair, "cwe_id", "UNKNOWN") or "UNKNOWN"
        cve = getattr(pair, "cve_id", "UNKNOWN") or "UNKNOWN"
        by_cwe[cwe][cve].append(pair)

    cwes = sorted(by_cwe)
    n_cwes = max(1, len(cwes))
    cwe_pair_counts = {c: sum(len(v) for v in by_cwe[c].values()) for c in cwes}
    total_pairs = sum(cwe_pair_counts.values()) or 1

    if mode == "balanced":
        if total and total > 0:
            base = max(1, int(total) // n_cwes)
        else:
            base = min(cwe_pair_counts.values())
        pair_quota = {c: base for c in cwes}
    else:  # proportional
        target_total = int(total) if total and total > 0 else total_pairs
        pair_quota = {
            c: max(1, round(target_total * cwe_pair_counts[c] / total_pairs))
            for c in cwes
        }

    sampled: list = []
    per_cwe: dict[str, dict[str, int]] = {}
    for cwe in cwes:
        groups = list(by_cwe[cwe].items())  # (cve_id, [pairs])
        rng.shuffle(groups)
        # Prefer multi-pair CVEs so each CWE keeps query-capable groups.
        groups.sort(key=lambda kv: len(kv[1]), reverse=True)

        quota = pair_quota[cwe]
        picked: list = []
        n_cve = 0
        for _cve, grp in groups:
            if n_cve < min_cves_per_cwe or len(picked) < quota:
                picked.extend(grp)
                n_cve += 1
            else:
                break
        sampled.extend(picked)
        per_cwe[cwe] = {"cves": n_cve, "pairs": len(picked)}

    rng.shuffle(sampled)

    info = {
        "mode": mode,
        "applied": True,
        "seed": seed,
        "min_cves_per_cwe": min_cves_per_cwe,
        "target_total": int(total) if total and total > 0 else None,
        "result_total_pairs": len(sampled),
        "result_total_cves": sum(v["cves"] for v in per_cwe.values()),
        "per_cwe": per_cwe,
    }
    return sampled, info
