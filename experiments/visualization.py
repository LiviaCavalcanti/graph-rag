import collections
import json
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── helpers ────────────────────────────────────────────────────────────

def _multi_variant(raw_data: dict) -> bool:
    return len({c['graph_variant'] for c in raw_data['cells']}) > 1


def _strategy_label(cell: dict, multi: bool) -> str:
    if multi:
        return f"{cell['embedder']} / {cell['graph_variant']} / {cell['backend']}"
    return f"{cell['embedder']} / {cell['backend']}"


def _palette(series) -> dict:
    vals = list(dict.fromkeys(series))  # unique, insertion-ordered
    return dict(zip(vals, sns.color_palette("husl", len(vals))))


# ── Dashboard 1: Performance (latency + embedding cost) ───────────────

def create_performance_dashboard(raw_data: dict) -> plt.Figure:
    """Embedding cost, index build, and query latency."""
    rows = []
    for cell in raw_data['cells']:
        rows.append({
            'embedder':       cell['embedder'],
            'backend':        cell['backend'],
            'graph_variant':  cell['graph_variant'],
            'embed_time_s':   cell['embed_time_s'],
            'index_build_s':  cell['index_build_s'],
            'latency_mean_ms': cell['query_latency']['mean_ms'],
            'latency_p95_ms': cell['query_latency']['p95_ms'],
            'pairwise_sim':   cell['space_stats']['mean_pairwise_sim'],
            'effective_dim':  cell['space_stats']['effective_dim'],
        })
    df = pd.DataFrame(rows)
    multi = _multi_variant(raw_data)
    palette = _palette(df['embedder'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Performance: Cost & Latency', fontsize=18, fontweight='bold')

    # 1. Embedding time
    sns.barplot(ax=axes[0, 0], data=df, x='embedder', y='embed_time_s',
                hue='embedder', palette=palette, legend=False)
    axes[0, 0].set_title('Embedding Time (s)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('')

    # 2. Latency by backend
    sns.boxplot(ax=axes[0, 1], data=df, x='backend', y='latency_mean_ms',
                hue='embedder', palette=palette)
    axes[0, 1].set_title('Mean Query Latency (ms) by Backend')
    axes[0, 1].set_yscale('log')

    # 3. Embedding space — mean pairwise similarity
    x_var = 'graph_variant' if multi else 'backend'
    sns.stripplot(ax=axes[1, 0], data=df, x=x_var, y='pairwise_sim',
                  hue='embedder', palette=palette, dodge=True, size=10, linewidth=1)
    axes[1, 0].set_title('Embedding Space: Mean Pairwise Similarity\n'
                          '(lower = more distinct representations)')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].axhline(0.95, color='red', ls='--', lw=1, alpha=0.6, label='collapse threshold')
    axes[1, 0].legend(fontsize=7)

    # 4. Build time vs p95 latency trade-off
    sns.scatterplot(ax=axes[1, 1], data=df, x='index_build_s', y='latency_p95_ms',
                    hue='embedder', style='backend', palette=palette, s=150)
    axes[1, 1].set_title('Trade-off: Index Build Time vs. P95 Latency')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


# ── Dashboard 2: Self-Retrieval (same-CVE retrieval) ──────────────────

def create_self_retrieval_dashboard(raw_data: dict) -> plt.Figure:
    """
    Self-retrieval: given the vulnerable graph, does the RAG retrieve
    the same CVE from the index?

    Metrics: Hit@1, Hit@5, Hit@10, MRR (per-query breakdown).
    k=10 precision intentionally omitted — it is dominated by set size.
    """
    multi = _multi_variant(raw_data)
    rows_per_query = []
    summary_rows   = []

    for cell in raw_data['cells']:
        label    = _strategy_label(cell, multi)
        embedder = cell['embedder']
        sr       = cell.get('self_retrieval', {})

        summary_rows.append({
            'strategy': label,
            'embedder': embedder,
            'Hit@1':    sr.get('hit@1', 0),
            'Hit@5':    sr.get('hit@5', 0),
            'Hit@10':   sr.get('hit@10', 0),
            'MRR':      sr.get('mrr', 0),
            'n':        sr.get('n', 0),
        })

        for q in sr.get('raw_queries', []):
            rows_per_query.append({
                'strategy': label,
                'embedder': embedder,
                'query_cwe': q.get('query_cwe', 'UNKNOWN'),
                'mrr':       q.get('mrr', 0),
                'hit':       float(q.get('hit', False)),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_q       = pd.DataFrame(rows_per_query)
    palette    = _palette(df_summary['embedder'])

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Self-Retrieval Metrics — "Given a vulnerable function, find the same CVE"\n'
                 'Higher is better. Evaluated on held-out augmented test set.',
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_hit1  = fig.add_subplot(gs[0, 0])
    ax_hit5  = fig.add_subplot(gs[0, 1])
    ax_hit10 = fig.add_subplot(gs[0, 2])
    ax_mrr   = fig.add_subplot(gs[1, :2])
    ax_cwe   = fig.add_subplot(gs[1, 2])

    # Hit@1 / Hit@5 / Hit@10 side-by-side bars
    for ax, metric, title in [
        (ax_hit1,  'Hit@1',  'Hit@1  (top-1 is correct CVE)'),
        (ax_hit5,  'Hit@5',  'Hit@5  (correct CVE in top 5)'),
        (ax_hit10, 'Hit@10', 'Hit@10 (correct CVE in top 10)'),
    ]:
        sns.barplot(ax=ax, data=df_summary, x='strategy', y=metric,
                    hue='embedder', palette=palette, legend=(ax is ax_hit1))
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)
    ax_hit1.legend(title='Embedder', fontsize=7, title_fontsize=8)

    # MRR horizontal bar
    df_mrr_sorted = df_summary.sort_values('MRR', ascending=True)
    ax_mrr.barh(df_mrr_sorted['strategy'], df_mrr_sorted['MRR'],
                color=[palette.get(e, 'steelblue') for e in df_mrr_sorted['embedder']])
    ax_mrr.set_title('Mean Reciprocal Rank (MRR)', fontsize=11)
    ax_mrr.set_xlabel('MRR  (1.0 = always retrieved at rank 1)')
    ax_mrr.set_xlim(0, max(df_mrr_sorted['MRR'].max() * 1.25, 0.1))
    for i, (_, row) in enumerate(df_mrr_sorted.iterrows()):
        ax_mrr.text(row['MRR'] + 0.005, i,
                    f"{row['MRR']:.3f}  (n={int(row['n'])})", va='center', fontsize=8)

    # Per-CWE MRR breakdown (best strategy only)
    if not df_q.empty and df_q['query_cwe'].nunique() > 1:
        top_strategy = df_summary.sort_values('MRR', ascending=False).iloc[0]['strategy']
        df_top = df_q[df_q['strategy'] == top_strategy]
        cwe_mrr = df_top.groupby('query_cwe')['mrr'].mean().sort_values()
        cwe_mrr.plot(kind='barh', ax=ax_cwe, color='steelblue')
        ax_cwe.set_title(f'MRR per CWE\n(best strategy: {top_strategy})', fontsize=9)
        ax_cwe.set_xlabel('MRR')
        ax_cwe.tick_params(axis='y', labelsize=7)
    else:
        ax_cwe.axis('off')

    return fig


# ── Dashboard 3: CWE Recall — group-level retrieval analysis ──────────

def create_cwe_recall_dashboard(raw_data: dict) -> plt.Figure:
    """
    CWE-group recall: given a query, what fraction of the index items
    with the same CWE class are retrieved in the top-k?

    Includes:
      - Per-CWE recall bars (sorted worst→best)
      - Misclassification heatmap for worst CWEs
      - Macro recall comparison across strategies
    """
    multi = _multi_variant(raw_data)

    # ── aggregate per-CWE recall and confusion across strategies ──
    # Use the best-MRR strategy for per-CWE breakdown, show all for macro
    all_strategy_macro = []
    best_cell          = max(raw_data['cells'],
                             key=lambda c: c.get('self_retrieval', {}).get('mrr', 0))

    for cell in raw_data['cells']:
        label = _strategy_label(cell, multi)
        cwr   = cell.get('cwe_recall', {})
        all_strategy_macro.append({
            'strategy':    label,
            'embedder':    cell['embedder'],
            'macro_recall': cwr.get('macro_avg', 0),
            'n_cwes':      cwr.get('n_cwes', 0),
            'n_queries':   cwr.get('n_queries', 0),
        })

    df_macro = pd.DataFrame(all_strategy_macro).sort_values('macro_recall', ascending=True)
    palette  = _palette(df_macro['embedder'])

    # Per-CWE breakdown from best cell
    best_cwr   = best_cell.get('cwe_recall', {})
    per_cwe    = best_cwr.get('per_cwe', {})
    best_label = _strategy_label(best_cell, multi)

    df_per_cwe = pd.DataFrame([
        {'cwe': k, 'recall': v['recall'], 'support': v['support']}
        for k, v in per_cwe.items()
    ]).sort_values('recall')

    # Build confusion matrix from raw queries (best cell)
    confusion: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for q in best_cwr.get('raw_queries', []):
        qcwe = q.get('query_cwe', '')
        if not qcwe:
            continue
        for r in q.get('retrieved', [])[:10]:
            rcwe = r.get('cwe_id', '')
            if rcwe and rcwe != qcwe:
                confusion[qcwe][rcwe] += 1

    # Identify worst 8 CWEs for the confusion heatmap
    worst_cwes = df_per_cwe.head(8)['cwe'].tolist()
    all_retrieved_cwes = sorted({
        cwe for q_cwe in worst_cwes
        for cwe in confusion.get(q_cwe, {})
    })
    if len(all_retrieved_cwes) > 12:
        # keep only the most frequent confusors
        total_counts: collections.Counter = collections.Counter()
        for q_cwe in worst_cwes:
            total_counts.update(confusion.get(q_cwe, {}))
        all_retrieved_cwes = [c for c, _ in total_counts.most_common(12)]

    # ── figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle('CWE-Group Recall — "Does the RAG retrieve same-class vulnerabilities?"\n'
                 'Recall@10: fraction of same-CWE index items found in top-10 results.',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)
    ax_macro   = fig.add_subplot(gs[0, :])
    ax_per_cwe = fig.add_subplot(gs[1, 0])
    ax_heatmap = fig.add_subplot(gs[1, 1])

    # ── macro recall comparison ────────────────────────────────────
    ax_macro.barh(df_macro['strategy'], df_macro['macro_recall'],
                  color=[palette.get(e, 'steelblue') for e in df_macro['embedder']])
    ax_macro.set_title('Macro-Average CWE Recall@10 per Strategy', fontsize=12)
    ax_macro.set_xlabel('Recall  (macro-avg across CWE classes)')
    ax_macro.set_xlim(0, min(1.05, max(df_macro['macro_recall'].max() * 1.35, 0.15)))
    for i, (_, row) in enumerate(df_macro.iterrows()):
        ax_macro.text(row['macro_recall'] + 0.003, i,
                      f"{row['macro_recall']:.3f}  "
                      f"(n_cwes={int(row['n_cwes'])}, n_queries={int(row['n_queries'])})",
                      va='center', fontsize=8)

    # ── per-CWE recall bar (best strategy) ────────────────────────
    if not df_per_cwe.empty:
        colors = ['#d62728' if r < 0.2 else '#ff7f0e' if r < 0.5 else '#2ca02c'
                  for r in df_per_cwe['recall']]
        bars = ax_per_cwe.barh(df_per_cwe['cwe'], df_per_cwe['recall'], color=colors)
        ax_per_cwe.set_title(f'Per-CWE Recall@10\n(best strategy: {best_label})', fontsize=10)
        ax_per_cwe.set_xlabel('Recall')
        ax_per_cwe.set_xlim(0, 1.15)
        ax_per_cwe.tick_params(axis='y', labelsize=7)
        for bar, (_, row) in zip(bars, df_per_cwe.iterrows()):
            ax_per_cwe.text(
                bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={int(row['support'])}", va='center', fontsize=6,
            )
        # legend
        from matplotlib.patches import Patch
        ax_per_cwe.legend(handles=[
            Patch(color='#d62728', label='recall < 0.2'),
            Patch(color='#ff7f0e', label='0.2 – 0.5'),
            Patch(color='#2ca02c', label='≥ 0.5'),
        ], fontsize=7, loc='lower right')

    # ── confusion heatmap for worst CWEs ──────────────────────────
    if worst_cwes and all_retrieved_cwes:
        mat = np.zeros((len(worst_cwes), len(all_retrieved_cwes)), dtype=int)
        for i, qcwe in enumerate(worst_cwes):
            for j, rcwe in enumerate(all_retrieved_cwes):
                mat[i, j] = confusion[qcwe].get(rcwe, 0)

        df_heat = pd.DataFrame(mat, index=worst_cwes, columns=all_retrieved_cwes)
        sns.heatmap(
            df_heat, ax=ax_heatmap, annot=True, fmt='d',
            cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'retrieval count'},
        )
        ax_heatmap.set_title(
            f'Misclassification Heatmap — Worst 8 CWEs\n'
            f'(cells = how often row-CWE query retrieved column-CWE instead)\n'
            f'Strategy: {best_label}',
            fontsize=9,
        )
        ax_heatmap.set_xlabel('Retrieved CWE (wrong class)', fontsize=9)
        ax_heatmap.set_ylabel('Query CWE (true class)', fontsize=9)
        ax_heatmap.tick_params(axis='x', rotation=40, labelsize=7)
        ax_heatmap.tick_params(axis='y', rotation=0, labelsize=7)
    else:
        ax_heatmap.axis('off')
        ax_heatmap.text(0.5, 0.5, 'No confusion data available',
                        ha='center', va='center', transform=ax_heatmap.transAxes)

    return fig


# ── Dashboard 4: Summary table ─────────────────────────────────────────

def create_summary_dashboard(raw_data: dict) -> plt.Figure:
    """One-row-per-cell numerical summary with all key metrics."""
    multi = _multi_variant(raw_data)
    rows  = []
    for cell in raw_data['cells']:
        sr  = cell.get('self_retrieval', {})
        cwr = cell.get('cwe_recall', {})
        ss  = cell.get('space_stats', {})
        row = {
            'Embedder':     cell['embedder'],
            'Backend':      cell['backend'],
            '# Queries':    sr.get('n', 0),
            'Hit@1':        f"{sr.get('hit@1', 0):.3f}",
            'Hit@5':        f"{sr.get('hit@5', 0):.3f}",
            'Hit@10':       f"{sr.get('hit@10', 0):.3f}",
            'MRR':          f"{sr.get('mrr', 0):.3f}",
            'CWE Rec@10':   f"{cwr.get('macro_avg', 0):.3f}",
            '# CWEs':       cwr.get('n_cwes', 0),
            'Eff. Dim':     f"{ss.get('effective_dim', 0):.1f}",
            'Mean Sim':     f"{ss.get('mean_pairwise_sim', 0):.3f}",
            'Embed (s)':    f"{cell.get('embed_time_s', 0):.1f}",
            'Lat p95 (ms)': f"{cell.get('query_latency', {}).get('p95_ms', 0):.1f}",
        }
        if multi:
            row['Graph'] = cell['graph_variant']
        rows.append(row)

    sort_cols = (['Embedder', 'Graph', 'Backend'] if multi
                 else ['Embedder', 'Backend'])
    df = pd.DataFrame(rows).sort_values(sort_cols)

    fig, ax = plt.subplots(figsize=(max(16, len(df.columns) * 1.4), max(4, len(df) * 0.6 + 2)))
    fig.suptitle('Full Results Summary', fontsize=16, fontweight='bold')
    ax.axis('off')

    col_labels = list(df.columns)
    tbl = ax.table(
        cellText   = df.values.tolist(),
        colLabels  = col_labels,
        loc        = 'center',
        cellLoc    = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    # header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Self-retrieval columns: light blue; CWE columns: light green
    sr_cols  = {'Hit@1', 'Hit@5', 'Hit@10', 'MRR', '# Queries'}
    cwe_cols = {'CWE Rec@10', '# CWEs'}
    for i in range(1, len(df) + 1):
        for j, col in enumerate(col_labels):
            if col in sr_cols:
                base = '#ddeeff'
            elif col in cwe_cols:
                base = '#ddffdd'
            else:
                base = '#f0f0f0' if i % 2 == 0 else 'white'
            tbl[i, j].set_facecolor(base)

    # column group labels above header
    ax.text(0.01, 0.97, '■ blue = self-retrieval  ■ green = CWE-group recall',
            transform=ax.transAxes, fontsize=8, va='top',
            color='#444', style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── entry point ────────────────────────────────────────────────────────

def generate_visualizations(raw_data: dict, output_dir: str) -> None:
    """Generate and save all dashboards to output_dir/visualizations/."""
    import os
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    sns.set_theme(style='whitegrid')

    dashboards = [
        ('dashboard_performance.png',    create_performance_dashboard),
        ('dashboard_self_retrieval.png', create_self_retrieval_dashboard),
        ('dashboard_cwe_recall.png',     create_cwe_recall_dashboard),
        ('dashboard_summary.png',        create_summary_dashboard),
    ]

    for filename, fn in dashboards:
        try:
            fig = fn(raw_data)
            fig.savefig(os.path.join(vis_dir, filename), dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  [warning] {filename} failed: {e}")
            plt.close('all')

    print(f"Visualizations saved → {vis_dir}")

    # Generate unified HTML dashboard if possible
    try:
        from experiments.dashboard import generate_html_dashboard
        generate_html_dashboard(output_dir)
    except Exception as e:
        print(f"  [warning] unified dashboard failed: {e}")


# ── backwards-compat stubs (used by existing tests/runner imports) ─────

def prepare_data(raw_data):
    rows = []
    for cell in raw_data['cells']:
        row = {
            'embedder': cell['embedder'],
            'backend': cell['backend'],
            'graph_variant': cell['graph_variant'],
            'embed_time_s': cell['embed_time_s'],
            'index_build_s': cell['index_build_s'],
            'latency_mean_ms': cell['query_latency']['mean_ms'],
            'latency_p95_ms': cell['query_latency']['p95_ms'],
            'pairwise_sim': cell['space_stats']['mean_pairwise_sim'],
            'mrr': cell['self_retrieval']['mrr'],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_quality_data(raw_data):
    rows = []
    for cell in raw_data['cells']:
        row = {
            'embedder': cell['embedder'],
            'variant': cell['graph_variant'],
            'backend': cell['backend'],
            'mean_sim': cell['space_stats']['mean_pairwise_sim'],
            'std_sim': cell['space_stats']['std_pairwise_sim'],
            'eff_dim': cell['space_stats']['effective_dim'],
            'mrr': cell['self_retrieval']['mrr'],
            'hit1': cell['self_retrieval']['hit@1'],
            'embed_time': cell['embed_time_s'],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    multi_variant = df['variant'].nunique() > 1
    df['strategy'] = df['embedder'] + '_' + df['variant'] if multi_variant else df['embedder']
    return df


def prepare_metrics_data(raw_data: dict) -> pd.DataFrame:
    all_variants  = {cell['graph_variant'] for cell in raw_data['cells']}
    multi_variant = len(all_variants) > 1
    rows = []
    for cell in raw_data['cells']:
        embedder      = cell['embedder']
        graph_variant = cell['graph_variant']
        backend       = cell['backend']
        label = (f"{embedder} / {graph_variant} / {backend}"
                 if multi_variant else f"{embedder} / {backend}")
        for q in cell.get('self_retrieval', {}).get('raw_queries', []):
            retrieved = q.get('retrieved', [])
            for k in [1, 5, 10]:
                top_k = retrieved[:k]
                tp    = sum(1 for r in top_k if r.get('cve_id') == q['query_cve'])
                rows.append({
                    'label': label, 'embedder': embedder,
                    'graph_variant': graph_variant, 'backend': backend,
                    'source': 'code_query', 'query_cve': q['query_cve'],
                    'query_cwe': q.get('query_cwe', ''), 'k': k,
                    'precision': tp / k,
                    'recall': float(tp > 0),
                    'mrr': q.get('mrr', 0.0),
                })
        for q in cell.get('cwe_recall', {}).get('raw_queries', []):
            retrieved = q.get('retrieved', [])
            cwe = q.get('query_cwe', '')
            for k in [1, 5, 10]:
                top_k    = retrieved[:k]
                same_cwe = sum(1 for r in top_k if r.get('cwe_id') == cwe)
                rows.append({
                    'label': label, 'embedder': embedder,
                    'graph_variant': graph_variant, 'backend': backend,
                    'source': 'cwe_group', 'query_cve': q['query_cve'],
                    'query_cwe': cwe, 'k': k,
                    'precision': same_cwe / k,
                    'recall': (q.get('recall', 0.0) if k == 10
                               else same_cwe / max(1, k)),
                    'mrr': 0.0,
                })
    return pd.DataFrame(rows)

    rows = []
    for cell in raw_data['cells']:
        # Flatten nested dictionaries
        row = {
            'embedder': cell['embedder'],
            'backend': cell['backend'],
            'graph_variant': cell['graph_variant'],
            'embed_time_s': cell['embed_time_s'],
            'index_build_s': cell['index_build_s'],
            'latency_mean_ms': cell['query_latency']['mean_ms'],
            'latency_p95_ms': cell['query_latency']['p95_ms'],
            'pairwise_sim': cell['space_stats']['mean_pairwise_sim'],
            'mrr': cell['self_retrieval']['mrr']
        }
        rows.append(row)
    return pd.DataFrame(rows)
