"""Text similarity metrics for code patch evaluation."""

from __future__ import annotations

import collections
import difflib
import math
import re

# ── helpers ──────────────────────────────────────────────────────────


def tokenize(code: str) -> list[str]:
    """Split code into identifier / operator / literal tokens."""
    return re.findall(r"[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\s]", code)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ── metrics ──────────────────────────────────────────────────────────


def exact_match(gen: str, ref: str) -> bool:
    return gen.strip() == ref.strip()


def normalised_exact_match(gen: str, ref: str) -> bool:
    """Whitespace-insensitive exact match."""
    return re.sub(r"\s+", " ", gen).strip() == re.sub(r"\s+", " ", ref).strip()


def sequence_matcher_ratio(gen: str, ref: str) -> float:
    """difflib SequenceMatcher ratio (char-level)."""
    return difflib.SequenceMatcher(None, gen, ref).ratio()


def line_level_ratio(gen: str, ref: str) -> float:
    """difflib SequenceMatcher on lines."""
    return difflib.SequenceMatcher(
        None, gen.splitlines(keepends=True), ref.splitlines(keepends=True)
    ).ratio()


def levenshtein_distance(a: str, b: str) -> int:
    """Character-level edit distance (Wagner–Fischer)."""
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def normalised_edit_distance(gen: str, ref: str) -> float:
    """Edit distance normalised by max length (0 = identical, 1 = nothing shared)."""
    d = levenshtein_distance(gen, ref)
    return d / max(len(gen), len(ref), 1)


def token_jaccard(gen: str, ref: str) -> float:
    """Jaccard similarity on code tokens."""
    tg = set(tokenize(gen))
    tr = set(tokenize(ref))
    if not tg and not tr:
        return 1.0
    return len(tg & tr) / len(tg | tr) if (tg | tr) else 0.0


def token_jaccard_multiset(gen: str, ref: str) -> float:
    """Jaccard on token multisets (accounts for frequency)."""
    cg = collections.Counter(tokenize(gen))
    cr = collections.Counter(tokenize(ref))
    intersection = sum((cg & cr).values())
    union = sum((cg | cr).values())
    return intersection / union if union else 1.0


def bleu_score(gen: str, ref: str, max_n: int = 4) -> float:
    """Corpus-level BLEU (single-reference) on code tokens, with brevity penalty."""
    gen_tokens = tokenize(gen)
    ref_tokens = tokenize(ref)

    if not ref_tokens:
        return 1.0 if not gen_tokens else 0.0
    if not gen_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        gen_ng = ngrams(gen_tokens, n)
        ref_ng = ngrams(ref_tokens, n)
        if not gen_ng:
            precisions.append(0.0)
            continue
        ref_counts = collections.Counter(ref_ng)
        clipped = sum(
            min(collections.Counter(gen_ng)[ng], ref_counts[ng]) for ng in set(gen_ng)
        )
        precisions.append(clipped / len(gen_ng))

    # avoid log(0)
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(gen_tokens)))
    return bp * math.exp(log_avg)


def codebleu_weighted(gen: str, ref: str) -> float:
    """Simplified CodeBLEU proxy: 0.5*BLEU + 0.25*token_jaccard + 0.25*line_ratio."""
    return (
        0.50 * bleu_score(gen, ref)
        + 0.25 * token_jaccard(gen, ref)
        + 0.25 * line_level_ratio(gen, ref)
    )


# ── diff details ─────────────────────────────────────────────────────


def compute_diff_details(gen: str, ref: str) -> dict:
    """Return structured diff information: hunks, changed line numbers, unified diff."""
    gen_lines = gen.splitlines(keepends=True)
    ref_lines = ref.splitlines(keepends=True)

    unified = list(
        difflib.unified_diff(
            ref_lines,
            gen_lines,
            fromfile="ground_truth",
            tofile="generated",
            lineterm="",
        )
    )

    # extract hunk positions
    hunks = []
    added_lines = []
    removed_lines = []
    ref_line = gen_line = 0

    for line in unified:
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                ref_line = int(m.group(1))
                gen_line = int(m.group(2))
                hunks.append({"ref_start": ref_line, "gen_start": gen_line})
        elif line.startswith("-") and not line.startswith("---"):
            removed_lines.append(ref_line)
            ref_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added_lines.append(gen_line)
            gen_line += 1
        else:
            ref_line += 1
            gen_line += 1

    return {
        "num_hunks": len(hunks),
        "hunks": hunks,
        "added_line_numbers_in_generated": added_lines,
        "removed_line_numbers_in_ground_truth": removed_lines,
        "total_added_lines": len(added_lines),
        "total_removed_lines": len(removed_lines),
        "unified_diff": "".join(unified)[:5000],  # cap size
    }
