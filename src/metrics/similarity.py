"""Text similarity metrics for code patch evaluation."""

from __future__ import annotations

import collections
import difflib
import math
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


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


# ── BERTScore ────────────────────────────────────────────────────────

# Lazy-loaded singleton so the model is loaded once per process.
_bertscore_model = None
_bertscore_tokenizer = None
_bertscore_device = None


def _get_bertscore_model(
    model_name: str = "microsoft/codebert-base",
) -> tuple:
    global _bertscore_model, _bertscore_tokenizer, _bertscore_device
    if _bertscore_model is None:
        _bertscore_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _bertscore_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _bertscore_model = (
            AutoModel.from_pretrained(model_name).to(_bertscore_device).eval()
        )
    return _bertscore_model, _bertscore_tokenizer, _bertscore_device


def bertscore_pair(
    gen: str,
    ref: str,
    model_name: str = "microsoft/codebert-base",
    baseline: float | None = None,
) -> dict[str, float]:
    """BERTScore (precision, recall, F1) for a single gen/ref pair.

    Uses CodeBERT by default (suitable for code patches).
    The model is loaded lazily and cached for the process lifetime.

    Special tokens ([CLS], [SEP], [PAD]) are excluded from scoring.

    If ``baseline`` is provided, raw scores are rescaled:
        rescaled = (raw - baseline) / (1 - baseline)
    This removes the high floor inherent in contextual embeddings.
    When baseline is None, a default of 0.85 is used (empirical
    baseline for CodeBERT on code patches).
    """
    if baseline is None:
        baseline = 0.85

    model, tokenizer, device = _get_bertscore_model(model_name)

    pred_enc = tokenizer(
        gen, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    ref_enc = tokenizer(
        ref, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        p_emb = model(**pred_enc).last_hidden_state[0]  # (Lp, D)
        r_emb = model(**ref_enc).last_hidden_state[0]  # (Lr, D)

    # Exclude special tokens: [CLS] at position 0 and [SEP] at the end.
    # For each sequence, the real tokens are positions 1..(len-1) exclusive
    # (last non-pad token is [SEP]).
    p_len = int(pred_enc["attention_mask"].sum().item())
    r_len = int(ref_enc["attention_mask"].sum().item())
    # Slice out [CLS] (idx 0) and [SEP] (last attended idx)
    p_emb = p_emb[1 : max(p_len - 1, 1)]
    r_emb = r_emb[1 : max(r_len - 1, 1)]

    if p_emb.shape[0] == 0 or r_emb.shape[0] == 0:
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }

    p_emb = torch.nn.functional.normalize(p_emb, dim=-1)
    r_emb = torch.nn.functional.normalize(r_emb, dim=-1)

    sim = p_emb @ r_emb.T  # (Lp, Lr)

    raw_precision = sim.max(dim=1).values.mean().item()
    raw_recall = sim.max(dim=0).values.mean().item()

    # Baseline rescaling: shift and stretch so random pairs score near 0
    def _rescale(raw: float) -> float:
        return max(0.0, (raw - baseline) / (1.0 - baseline))

    precision = _rescale(raw_precision)
    recall = _rescale(raw_recall)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "bertscore_precision": round(precision, 4),
        "bertscore_recall": round(recall, 4),
        "bertscore_f1": round(f1, 4),
    }


def bertscore(
    preds: list[str],
    refs: list[str],
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 8,
) -> dict:
    """
    BERTScore semantic similarity between predicted and reference texts.

    For each (pred, ref) pair, computes token-level cosine similarities
    between contextual embeddings and returns precision, recall, and F1.

    Uses CodeBERT by default since the project deals with code patches.

    Returns dict with keys: precision, recall, f1 (each a list[float]),
    and their mean counterparts: mean_precision, mean_recall, mean_f1.
    """
    pairs = list(zip(preds, refs))
    if not pairs:
        return {
            "precision": [],
            "recall": [],
            "f1": [],
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_precision, all_recall, all_f1 = [], [], []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_preds = [p for p, _ in batch]
        batch_refs = [r for _, r in batch]

        pred_enc = tokenizer(
            batch_preds,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        ref_enc = tokenizer(
            batch_refs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            pred_embs = model(**pred_enc).last_hidden_state  # (B, Lp, D)
            ref_embs = model(**ref_enc).last_hidden_state  # (B, Lr, D)

        # Normalize embeddings for cosine similarity
        pred_embs = torch.nn.functional.normalize(pred_embs, dim=-1)
        ref_embs = torch.nn.functional.normalize(ref_embs, dim=-1)

        # Build masks that exclude [PAD] tokens
        pred_mask = pred_enc["attention_mask"].bool()  # (B, Lp)
        ref_mask = ref_enc["attention_mask"].bool()  # (B, Lr)

        for j in range(len(batch_preds)):
            p_emb = pred_embs[j][pred_mask[j]]  # (lp, D)
            r_emb = ref_embs[j][ref_mask[j]]  # (lr, D)

            # Cosine similarity matrix: (lp, lr)
            sim = p_emb @ r_emb.T

            # Precision: for each pred token, max sim to any ref token
            precision = sim.max(dim=1).values.mean().item()
            # Recall: for each ref token, max sim to any pred token
            recall = sim.max(dim=0).values.mean().item()
            # F1
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            all_precision.append(round(precision, 4))
            all_recall.append(round(recall, 4))
            all_f1.append(round(f1, 4))

    return {
        "precision": all_precision,
        "recall": all_recall,
        "f1": all_f1,
        "mean_precision": round(float(np.mean(all_precision)), 4),
        "mean_recall": round(float(np.mean(all_recall)), 4),
        "mean_f1": round(float(np.mean(all_f1)), 4),
    }

