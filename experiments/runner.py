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
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field, asdict

from src.rag.index import FAISSIndex
from src.rag.hnsw import HNSWIndex
from src.rag.retriever import Retriever
from src.embeddings import build_embedders
from experiments.metrics import (
    self_retrieval_metrics,
    leave_one_out_metrics,
    cwe_group_recall,
    embedding_space_stats,
)
from experiments.visualization import generate_visualizations


OUTPUT_DIR = Path('experiments/output')

BACKEND_REGISTRY = {
    'faiss_flat': FAISSIndex,
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


def _run_code_query_eval(
    pairs: list,
    embeddings: np.ndarray,
    retriever,
    embedder,
    ks: list[int],
) -> dict:


    from collections import defaultdict

    hits = defaultdict(int)
    mrrs = []
    n=0

    for i, pair in enumerate(pairs):
        if pair.meta.get("dataset") == "autopatch" and pair.meta.get("variant") != "original":
            query_graph = pair.G_before
        else:
            query_graph = pair.G_vuln
        query_vec = embedder.embed_one(query_graph)
        if np.linalg.norm(query_vec) < 1e-6:
            continue
        results = retriever.query(query_vec, top_k=max(ks))
        for k in ks:
            hits[k] += int(any(r['cve_id'] == pair.cve_id for r in results[:k]))
        
        mrrs.append(next(
            (1.0 / (j + 1) for j, r in enumerate(results) if r['cve_id'] == pair.cve_id),
            0.0
        ))

        n+=1

    if n== 0:
        return {'n': 0}
    
    return {
        **{f'hit@{k}': hits[k] / n for k in ks},
        'mrr': float(np.mean(mrrs)),
        'n': n
    }


def _measure_latency(retriever, embeddings: np.ndarray, n_queries: int = 200) -> dict:
    """Sample query latencies in milliseconds."""
    idx     = np.random.choice(len(embeddings), min(n_queries, len(embeddings)), replace=False)
    samples = embeddings[idx]
    times   = []
    for vec in samples:
        t0 = time.perf_counter()
        retriever.query(vec, top_k=10)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'mean_ms': float(np.mean(times)),
    }


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

    print(f"\n{'='*60}")
    print(f"Experiment run: {run_id}")
    print(f"Samples: {len(pairs)}  |  output: {run_dir}")
    print(f"{'='*60}")

    embedders     = build_embedders(cfg)
    backend_names = list(BACKEND_REGISTRY.keys())
    graph_variants = ['G_vuln', 'G_before']

    cells: list[CellResult] = []
    for p in pairs:
        print(p.cve_id,
            p.G_vuln.number_of_nodes(),
            p.G_before.number_of_nodes(),
            len(set(attr.get('labelV') for _, attr in p.G_vuln.nodes(data=True))))
            
    for embedder, graph_variant in product(embedders, graph_variants):

        # ── embed all samples ────────────────────────────────────────
        print(f"\n  [{embedder.name} / {graph_variant}] embedding {len(pairs)} samples...")
        t0   = time.perf_counter()
        graphs = [
            getattr(p, graph_variant) for p in pairs
        ]
        embeddings = embedder.embed_many(graphs)   # (N, dim)
        embed_time = time.perf_counter() - t0
        print(f"    done in {embed_time:.1f}s")

        # intrinsic stats — same regardless of backend
        space_stats = embedding_space_stats(embeddings)
        print(f"    effective_dim={space_stats['effective_dim']:.1f}  "
              f"mean_sim={space_stats['mean_pairwise_sim']:.3f}")

        meta_list = [
            {
                'cve_id':    p.cve_id,
                'cwe_id':    p.cwe_id,
                'func_name': p.func_name,
                'project':   p.project,
                **p.meta,
            }
            for p in pairs
        ]

        for backend_name in backend_names:

            print(f"  [{embedder.name} / {graph_variant} / {backend_name}]")

            # ── build index ──────────────────────────────────────────
            t0    = time.perf_counter()
            index = _build_index(backend_name, embedder.dim, run_dir, embedder.name, graph_variant)
            for pair, vec, _ in zip(pairs, embeddings, meta_list):
                index.add(pair, vec, embedder.name)
            index.save()
            build_time = time.perf_counter() - t0

            index.load()
            retriever = Retriever(index, top_k=max(ks))

            # ── latency ──────────────────────────────────────────────
            latency = _measure_latency(retriever, embeddings)
            print(f"    latency p50={latency['p50_ms']:.2f}ms  p99={latency['p99_ms']:.2f}ms")

            # ── self-retrieval ───────────────────────────────────────
            sr = _run_code_query_eval(pairs, embeddings, retriever, embedder, ks=ks)
            print(sr)
            if sr.get('n') == 0:
                print(f"    code-query hit@1=NaN  mrr=-1")
            else:
                print(f"    code-query hit@1={sr['hit@1']:.3f}  mrr={sr['mrr']:.3f}")

            # ── CWE group recall ─────────────────────────────────────
            cwr = cwe_group_recall(embeddings, meta_list, retriever, top_k=max(ks))
            print(f"    CWE recall macro={cwr['macro_avg']:.3f}  n_cwes={cwr['n_cwes']}")

            # ── leave-one-out (optional — slow for large N) ──────────
            loo = {}
            if run_leave_one_out and len(pairs) <= 1000:
                print(f"    running leave-one-out ({len(pairs)} iterations)...")
                loo = leave_one_out_metrics(
                    embeddings  = embeddings,
                    metadata    = meta_list,
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
                n_samples     = len(pairs),
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
            'datasets': list({p.meta.get('dataset', '?') for p in pairs}),
            'cwe_ids':  list({p.cwe_id for p in pairs if p.cwe_id}),
            'projects': list({p.project for p in pairs}),
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