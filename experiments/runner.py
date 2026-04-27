"""
Experiment runner — compares embedders x RAG backends as a grid.

Each cell in the grid produces:
  - embedding space stats (intrinsic, no labels)
  - self-retrieval metrics  (hit@k, MRR)
  - CWE-group recall        (clustering quality)
  - leave-one-out metrics   (most honest, slowest)
  - query latency           (p50, p95, p99)

Results are written to experiments/output/<run_id>/results.json
One run = one dataset snapshot. Multiple runs can be compared later.
"""

import json
import time
import uuid
import random
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import logging

from src.rag.index import FAISSIndex
from src.rag.hnsw import HNSWIndex
from src.rag.retriever import Retriever
from src.embeddings import build_embedders
from src.metrics.metrics import (
    leave_one_out_metrics,
    embedding_space_stats,
    measure_latency,
)
from src.metrics.retrieval_eval import (
    code_query_eval,
    cross_cwe_recall,
)
from experiments.visualization import generate_visualizations


OUTPUT_DIR = Path('experiments/output')

BACKEND_REGISTRY = {
    # 'faiss_flat': FAISSIndex,
    'hnsw':       HNSWIndex,
}


@dataclass
class CellResult:
    """Result for one (embedder, backend) cell."""
    embedder:        str
    backend:         str
    graph_variant:   str                  # 'G_vuln' or 'G_before'
    n_samples:       int
    embed_time_s:    float                # total embedding time
    index_build_s:   float
    query_latency:   dict                 # p50/p95/p99 in ms
    space_stats:     dict
    self_retrieval:  dict
    cwe_recall:      dict
    leave_one_out:   dict = field(default_factory=dict)  # skipped if too slow


@dataclass
class ExperimentResult:
    run_id:       str
    timestamp:    str
    config:       dict
    dataset_info: dict
    cells:        list[CellResult]


def _is_original_pair(pair) -> bool:
    """Determine if a pair is from the original dataset vs augmented, based on metadata."""
    return pair.meta.get('variant') == 'original'


def _split_by_variant(pairs: list) -> tuple[list, list]:
    """Split pairs into original vs augmented based on metadata."""
    original = [p for p in pairs if _is_original_pair(p)]
    augmented = [p for p in pairs if not _is_original_pair(p)]
    return original, augmented


def _stratified_split_pairs(
    pairs: list,
    test_ratio: float,
    seed: int,
) -> tuple[list, list]:
    """
    Split pairs into train/test while preserving CWE distribution as much as possible.
    Falls back to a random split when stratification is not feasible.
    """
    if not pairs:
        return [], []

    if len(pairs) == 1:
        return pairs[:], []

    test_ratio = max(0.0, min(0.9, test_ratio))
    rng = random.Random(seed)

    by_cwe = defaultdict(list)
    known_cwe_count = 0
    for p in pairs:
        cwe = p.cwe_id if p.cwe_id and p.cwe_id != 'UNKNOWN' else '__UNKNOWN__'
        by_cwe[cwe].append(p)
        if p.cwe_id and p.cwe_id != 'UNKNOWN':
            known_cwe_count += 1
    logging.info(f"Stratified split: {len(by_cwe)} CWE groups ({known_cwe_count} pairs with known CWE)")

    train, test = [], []
    for cwe_pairs in by_cwe.values():
        items = cwe_pairs[:]
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio))
        if n > 1:
            n_test = max(1, min(n - 1, n_test))
        else:
            n_test = 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    # global guardrails to keep both sets non-empty when possible
    if not test and len(pairs) > 1:
        rng.shuffle(train)
        test.append(train.pop())

    if not train and len(pairs) > 1:
        rng.shuffle(test)
        train.append(test.pop())

    return train, test


def _sample_pairs(
    pairs: list,
    keep_ratio: float,
    seed: int,
) -> list:
    """Uniform random downsampling used for augmented-train contribution."""
    if not pairs:
        return []
    keep_ratio = max(0.0, min(1.0, keep_ratio))
    if keep_ratio >= 1.0:
        return pairs[:]
    if keep_ratio <= 0.0:
        return []
    rng = random.Random(seed)
    items = pairs[:]
    rng.shuffle(items)
    k = max(1, int(round(len(items) * keep_ratio)))
    return items[:k]


def _build_split_plan(pairs: list, cfg: dict) -> tuple[list, list, dict]:
    """
    Build index/query sets for experiments.

    Default behavior keeps backwards compatibility (all pairs in both sets).
    With split enabled, index is typically real + sampled augmented train,
    and query set is held-out augmented test.
    """
    split_cfg = (cfg or {}).get('experiment', {}).get('split', {})
    split_enabled = bool(split_cfg.get('enabled', False))

    if not split_enabled:
        return pairs[:], pairs[:], {
            'enabled': False,
            'index_n': len(pairs),
            'query_n': len(pairs),
            'mode': 'all_vs_all',
        }

    seed = int(split_cfg.get('seed', 42))
    test_ratio = float(split_cfg.get('test_ratio', 0.2))
    use_stratified = bool(split_cfg.get('stratified', True))
    include_real_in_index = bool(split_cfg.get('include_real_in_index', True))
    # fraction of the augmented training set to keep in the index; ablation on how much augmented data helps
    # for fractions between 0 and 1, we randomly sample a subset of the augmented training pairs
    aug_train_ratio = float(split_cfg.get('augmented_train_ratio', 1.0))
    query_source = str(split_cfg.get('query_source', 'augmented_test'))

    real_pairs, aug_pairs = _split_by_variant(pairs)

    # Stratified split on augmented pairs; real pairs are typically too few to split meaningfully, so we can optionally include them all in the index for training signal.
    if use_stratified:
        aug_train, aug_test = _stratified_split_pairs(aug_pairs, test_ratio=test_ratio, seed=seed)
    else:
        rng = random.Random(seed)
        shuffled = aug_pairs[:]
        rng.shuffle(shuffled)
        cut = int(round(len(shuffled) * (1.0 - max(0.0, min(0.9, test_ratio)))))
        aug_train, aug_test = shuffled[:cut], shuffled[cut:]

    # include on training set the augmented pairs that are not held out for testing, but we can optionally sample only a fraction of them to see how it impacts results
    aug_train_kept = _sample_pairs(aug_train, keep_ratio=aug_train_ratio, seed=seed + 13)

    # Build index/query sets according to config. Default is all pairs in both sets for maximum training signal, but with split enabled we typically want to hold out the augmented test set as queries and keep the real + augmented train pairs in the index.
    index_pairs = []
    if include_real_in_index:
        index_pairs.extend(real_pairs)
    index_pairs.extend(aug_train_kept)

    if query_source == 'augmented_test':
        query_pairs = aug_test
    elif query_source == 'all_test':
        real_train, real_test = _stratified_split_pairs(real_pairs, test_ratio=test_ratio, seed=seed + 31)
        # keep real_train available for future tuning; index currently configured above
        _ = real_train
        query_pairs = aug_test + real_test
    elif query_source == 'augmented_train':
        query_pairs = aug_train_kept
    else:
        query_pairs = aug_test

    # Robust fallback to avoid empty index/query sets.
    if not index_pairs:
        index_pairs = real_pairs[:] if real_pairs else aug_train_kept[:]
    if not query_pairs:
        query_pairs = aug_test[:] if aug_test else index_pairs[:]

    split_info = {
        'enabled': True,
        'seed': seed,
        'stratified': use_stratified,
        'test_ratio': test_ratio,
        'query_source': query_source,
        'include_real_in_index': include_real_in_index,
        'augmented_train_ratio': aug_train_ratio,
        'counts': {
            'total': len(pairs),
            'real_total': len(real_pairs),
            'aug_total': len(aug_pairs),
            'aug_train_total': len(aug_train),
            'aug_train_used': len(aug_train_kept),
            'aug_test_total': len(aug_test),
            'index_total': len(index_pairs),
            'query_total': len(query_pairs),
        },
    }
    return index_pairs, query_pairs, split_info


def _build_index(
    backend_name: str,
    dim:          int,
    run_dir:      Path,
    embedder_name: str,
    graph_variant: str,
) -> FAISSIndex | HNSWIndex:
    stem = f"{embedder_name}__{graph_variant}"
    return BACKEND_REGISTRY[backend_name](
        dim           = dim,
        index_path    = str(run_dir / f'{stem}__{backend_name}.index'),
        metadata_path = str(run_dir / f'{stem}__{backend_name}_meta.json'),
    )


def run_experiment(
    pairs:            list,        # list[FunctionPair]
    cfg:              dict,
    run_leave_one_out: bool = False,
    ks:               list[int] = [1, 5, 10],
    output_dir:       Path = OUTPUT_DIR,
) -> ExperimentResult:

    run_id  = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    index_pairs, query_pairs, split_info = _build_split_plan(pairs, cfg)

    print(f"\n{'='*60}")
    print(f"Experiment run: {run_id}")
    print(f"Samples total: {len(pairs)}  |  index: {len(index_pairs)}  |  query: {len(query_pairs)}")
    if split_info.get('enabled'):
        counts = split_info.get('counts', {})
        print(
            "Split enabled: "
            f"real={counts.get('real_total', 0)}  "
            f"aug_train_used={counts.get('aug_train_used', 0)}  "
            f"aug_test={counts.get('aug_test_total', 0)}"
        )
    print(f"{'='*60}")

    embedders     = build_embedders(cfg)
    backend_names = list(BACKEND_REGISTRY.keys())
    graph_variants = ['G_vuln']  # ablation on which graph to embed

    cells: list[CellResult] = []
    for embedder, graph_variant in product(embedders, graph_variants):

        # embed all samples
        print(f"\n  [{embedder.name} / {graph_variant}] embedding index={len(index_pairs)} query={len(query_pairs)}...")
        t0 = time.perf_counter()
        index_graphs = [getattr(p, graph_variant) for p in index_pairs]
        index_embeddings = embedder.embed_many(index_graphs)  # (N, dim)
        embed_time = time.perf_counter() - t0
        print(f"    done in {embed_time:.1f}s")

        # intrinsic stats are computed on index vectors
        space_stats = embedding_space_stats(index_embeddings)
        print(f"    effective_dim={space_stats['effective_dim']:.1f}  "
              f"mean_sim={space_stats['mean_pairwise_sim']:.3f}")

        index_meta_list = [
            {
                'cve_id':    p.cve_id,
                'cwe_id':    p.cwe_id,
                'func_name': p.func_name,
                'project':   p.project,
                **p.meta,
            }
            for p in index_pairs
        ]

        for backend_name in backend_names:

            print(f"  [{embedder.name} / {graph_variant} / {backend_name}]")

            # ── build index ──────────────────────────────────────────
            t0    = time.perf_counter()
            index_dir = run_dir / 'indices'
            index_dir.mkdir(exist_ok=True)
            index = _build_index(backend_name, embedder.dim, index_dir, embedder.name, graph_variant)
            for pair, vec, _ in zip(index_pairs, index_embeddings, index_meta_list):
                index.add(pair, vec, embedder.name)
            index.save()
            build_time = time.perf_counter() - t0

            index.load()
            retriever = Retriever(index, top_k=max(ks))

            # ── latency ──────────────────────────────────────────────
            latency = measure_latency(retriever, index_embeddings)
            print(f"    latency p50={latency['p50_ms']:.2f}ms  p99={latency['p99_ms']:.2f}ms")

            # ── self-retrieval ───────────────────────────────────────
            sr = code_query_eval(query_pairs, retriever, embedder, ks=ks)
            # Omitting self-retrieval for its lengthy raw query logs, but they are available in the CellResult for later analysis.
            # print(sr)
            if sr.get('n') == 0:
                print(f"    code-query hit@1=NaN  mrr=-1")
            else:
                print(f"    code-query hit@1={sr['hit@1']:.3f}  mrr={sr['mrr']:.3f}")

            # ── CWE group recall ─────────────────────────────────────
            cwr = cross_cwe_recall(
                query_pairs=query_pairs,
                retriever=retriever,
                embedder=embedder,
                index_metadata=index.metadata,
                top_k=max(ks),
            )
            print(f"    CWE recall macro={cwr['macro_avg']:.3f}  n_cwes={cwr['n_cwes']}")

            # ── leave-one-out (optional — slow for large N) ──────────
            loo = {}
            if run_leave_one_out and not split_info.get('enabled') and len(index_pairs) <= 1000:
                print(f"    running leave-one-out ({len(index_pairs)} iterations)...")
                loo = leave_one_out_metrics(
                    embeddings  = index_embeddings,
                    metadata    = index_meta_list,
                    index_class = BACKEND_REGISTRY[backend_name],
                    index_kwargs = dict(
                        dim           = embedder.dim,
                        index_path    = str(run_dir / '_loo_tmp.index'),
                        metadata_path = str(run_dir / '_loo_tmp_meta.json'),
                    ),
                    ks = ks,
                )
                print(f"    LOO hit@1={loo.get('hit@1', 0):.3f}  mrr={loo.get('mrr', 0):.3f}")

            cells.append(CellResult(
                embedder      = embedder.name,
                backend       = backend_name,
                graph_variant = graph_variant,
                n_samples     = len(index_pairs),
                embed_time_s  = embed_time,
                index_build_s = build_time,
                query_latency = latency,
                space_stats   = space_stats,
                self_retrieval= sr,
                cwe_recall    = cwr,
                leave_one_out = loo,
            ))

    result = ExperimentResult(
        run_id       = run_id,
        timestamp    = datetime.now(timezone.utc).isoformat(),
        config       = cfg,
        dataset_info = {
            'n_pairs':  len(pairs),
            'n_index_pairs': len(index_pairs),
            'n_query_pairs': len(query_pairs),
            'datasets': list({p.meta.get('dataset', '?') for p in pairs}),
            'cwe_ids':  list({p.cwe_id for p in pairs if p.cwe_id}),
            'projects': list({p.project for p in pairs}),
            'split': split_info,
        },
        cells = cells,
    )

    # ── write results ────────────────────────────────────────────────
    out_path = run_dir / 'results.json'
    out_path.write_text(json.dumps(
        {
            'run_id':       result.run_id,
            'timestamp':    result.timestamp,
            'config':       result.config,
            'dataset_info': result.dataset_info,
            'cells':        [asdict(c) for c in result.cells],
        },
        indent=2,
    ))
    print(f"\nResults written → {out_path}")

    # ── write comparison summary ─────────────────────────────────────
    _write_summary(result, run_dir)

    # ── generate visualizations ──────────────────────────────────────
    raw = json.loads(out_path.read_text())
    generate_visualizations(raw, str(run_dir))

    return result


def _write_summary(result: ExperimentResult, run_dir: Path):
    """
    Flat summary table — easy to load into pandas for plotting.
    Each row is one (embedder, backend, graph_variant) cell.
    """
    rows = []
    for c in result.cells:
        rows.append({
            'run_id':           result.run_id,
            'timestamp':        result.timestamp,
            'embedder':         c.embedder,
            'backend':          c.backend,
            'graph_variant':    c.graph_variant,
            'n_samples':        c.n_samples,
            'embed_time_s':     c.embed_time_s,
            'index_build_s':    c.index_build_s,
            'latency_p50_ms':   c.query_latency['p50_ms'],
            'latency_p99_ms':   c.query_latency['p99_ms'],
            'effective_dim':    c.space_stats['effective_dim'],
            'mean_pairwise_sim':c.space_stats['mean_pairwise_sim'],
            'sr_hit@1':         c.self_retrieval.get('hit@1', 0),
            'sr_hit@5':         c.self_retrieval.get('hit@5', 0),
            'sr_hit@10':        c.self_retrieval.get('hit@10', 0),
            'sr_mrr':           c.self_retrieval.get('mrr', 0),
            'cwe_recall_macro': c.cwe_recall.get('macro_avg', 0),
            'cwe_n_groups':     c.cwe_recall.get('n_cwes', 0),
            'loo_hit@1':        c.leave_one_out.get('hit@1', 0),
            'loo_mrr':          c.leave_one_out.get('mrr', 0),
        })

    summary_path = run_dir / 'summary.json'
    summary_path.write_text(json.dumps(rows, indent=2))

    # also append to a global runs registry so you can compare across runs
    registry_path = run_dir.parent / 'all_runs.json'
    existing = json.loads(registry_path.read_text()) if registry_path.exists() else []
    existing.extend(rows)
    registry_path.write_text(json.dumps(existing, indent=2))

    print(f"Summary appended → {registry_path}")