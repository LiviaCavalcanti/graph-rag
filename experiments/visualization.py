import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Flatten Data
def prepare_data(raw_data):
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

def create_dashboard(df):
    # Set style and consistent color palette for Embedders
    sns.set_theme(style="whitegrid")
    embedders = df['embedder'].unique()
    palette = dict(zip(embedders, sns.color_palette("Set2", len(embedders))))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment Results Dashboard: Embedder Performance & Quality', fontsize=20)

    # Plot 1: Embedding Time (s) - Primary Efficiency Metric
    sns.barplot(ax=axes[0, 0], data=df, x='embedder', y='embed_time_s', hue='embedder', palette=palette, legend=False)
    axes[0, 0].set_title('Total Embedding Time (seconds) - Log Scale')
    axes[0, 0].set_yscale('log') # Handle small values

    # Plot 2: Query Latency (ms) - Comparison across Backends
    sns.boxplot(ax=axes[0, 1], data=df, x='backend', y='latency_mean_ms', hue='embedder', palette=palette)
    axes[0, 1].set_title('Mean Query Latency (ms) by Backend')
    axes[0, 1].set_yscale('log')

    # Plot 3: Space Stats - Mean Pairwise Similarity
    # This helps interpret if embeddings are collapsing or well-distributed
    sns.stripplot(ax=axes[1, 0], data=df, x='graph_variant', y='pairwise_sim', hue='embedder', 
                  palette=palette, dodge=True, size=10, linewidth=1)
    axes[1, 0].set_title('Space Distribution: Mean Pairwise Similarity')
    axes[1, 0].set_ylim(0, 1.1)

    # Plot 4: Index Build Time vs Latency (Trade-off)
    sns.scatterplot(ax=axes[1, 1], data=df, x='index_build_s', y='latency_p95_ms', 
                    hue='embedder', style='backend', palette=palette, s=150)
    axes[1, 1].set_title('Trade-off: Index Build Time vs. P95 Latency')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def prepare_quality_data(raw_data):
    rows = []
    for cell in raw_data['cells']:
        row = {
            'embedder': cell['embedder'],
            'variant': cell['graph_variant'],
            'backend': cell['backend'],
            'strategy': f"{cell['embedder']}_{cell['graph_variant']}",
            'mean_sim': cell['space_stats']['mean_pairwise_sim'],
            'std_sim': cell['space_stats']['std_pairwise_sim'],
            'eff_dim': cell['space_stats']['effective_dim'],
            'mrr': cell['self_retrieval']['mrr'],
            'hit1': cell['self_retrieval']['hit@1'],
            'embed_time': cell['embed_time_s']
        }
        rows.append(row)
    return pd.DataFrame(rows)

def create_quality_dashboard(df):
    sns.set_theme(style="white")
    palette = dict(zip(df['embedder'].unique(), sns.color_palette("husl", len(df['embedder'].unique()))))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Strategy Quality & Correctness Analysis', fontsize=22, fontweight='bold')

    # 1. Correctness: Similarity vs. Variation (The "Collapse" Zone)
    # Strategies in the top-left are "collapsed" (high similarity, low variation)
    # sns.scatterplot(ax=axes[0, 0], data=df, x='mean_sim', y='std_sim', hue='embedder', 
    #                 size='eff_dim', sizes=(50, 400), palette=palette, alpha=0.7)
    # axes[0, 0].set_title('Embedding Space Health: Similarity vs. Variation', fontsize=14)
    # axes[0, 0].axvline(0.95, color='red', linestyle='--', alpha=0.5, label='Collapse Threshold')
    # axes[0, 0].legend(title="Embedder (Size=Eff. Dim)")

    sns.barplot(ax=axes[0, 0], data=df, x='backend', y='hit1', hue='embedder', palette=palette)
    axes[0, 0].set_title('Retrieval Hit@1: Consistency Check', fontsize=14)
    axes[0, 0].set_ylim(0, 1)

    # 2. Value: Retrieval Hit@1 by Strategy
    sns.barplot(ax=axes[0, 1], data=df, x='strategy', y='hit1', hue='embedder', palette=palette)
    axes[0, 1].set_title('Retrieval Success (Hit@1)', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Representational Power: Effective Dimension
    # Is the embedding space actually utilizing its dimensions?
    sns.boxplot(ax=axes[1, 0], data=df, x='embedder', y='eff_dim', palette=palette, hue='embedder')
    axes[1, 0].set_title('Effective Dimensionality (Higher = More Distinct)', fontsize=14)

    # 4. ROI: Embedding Cost vs. Retrieval MRR
    sns.scatterplot(ax=axes[1, 1], data=df, x='embed_time', y='mrr', hue='embedder', 
                    style='variant', s=200, palette=palette)
    axes[1, 1].set_title('Return on Investment: Time vs. MRR', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def prepare_metrics_data(raw_data: dict):
    """
    Flatten per-sample raw_queries from self_retrieval and cwe_recall into
    a DataFrame suitable for computing and plotting recall, precision, and MRR.
    """
    rows = []
    for cell in raw_data['cells']:
        label = f"{cell['embedder']} / {cell['graph_variant']} / {cell['backend']}"
        embedder      = cell['embedder']
        graph_variant = cell['graph_variant']
        backend       = cell['backend']

        # ── per-query retrieval rows (from code-query eval) ─────────
        for q in cell.get('self_retrieval', {}).get('raw_queries', []):
            retrieved = q.get('retrieved', [])
            for k in [1, 5, 10]:
                top_k  = retrieved[:k]
                tp     = sum(1 for r in top_k if r.get('cve_id') == q['query_cve'])
                rows.append({
                    'label':        label,
                    'embedder':     embedder,
                    'graph_variant': graph_variant,
                    'backend':      backend,
                    'source':       'code_query',
                    'query_cve':    q['query_cve'],
                    'query_cwe':    q.get('query_cwe', ''),
                    'k':            k,
                    'precision':    tp / k,
                    'recall':       float(tp > 0),   # binary: did we find it?
                    'mrr':          q.get('mrr', 0.0),
                })

        # ── per-query CWE-group recall rows ─────────────────────────
        for q in cell.get('cwe_recall', {}).get('raw_queries', []):
            retrieved = q.get('retrieved', [])
            cwe       = q.get('query_cwe', '')
            for k in [1, 5, 10]:
                top_k    = retrieved[:k]
                same_cwe = sum(1 for r in top_k if r.get('cwe_id') == cwe)
                rows.append({
                    'label':        label,
                    'embedder':     embedder,
                    'graph_variant': graph_variant,
                    'backend':      backend,
                    'source':       'cwe_group',
                    'query_cve':    q['query_cve'],
                    'query_cwe':    cwe,
                    'k':            k,
                    'precision':    same_cwe / k,
                    'recall':       q.get('recall', 0.0) if k == 10 else same_cwe / max(1, k),
                    'mrr':          0.0,   # not applicable for CWE grouping
                })

    return pd.DataFrame(rows)


def create_metrics_dashboard(df: pd.DataFrame, raw_data: dict) -> plt.Figure:
    """
    Four-panel figure:
      1. Recall@k  grouped by embedder (code-query source)
      2. Precision@k grouped by embedder (code-query source)
      3. MRR grouped by embedder / graph_variant
      4. Summary table (one row per label)
    """
    sns.set_theme(style="whitegrid")
    palette = dict(zip(df['embedder'].unique(), sns.color_palette("husl", len(df['embedder'].unique()))))

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Retrieval Metrics: Recall, Precision & MRR', fontsize=20, fontweight='bold')

    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.35)
    ax_recall    = fig.add_subplot(gs[0, 0])
    ax_precision = fig.add_subplot(gs[0, 1])
    ax_mrr       = fig.add_subplot(gs[1, :])
    ax_table     = fig.add_subplot(gs[2, :])

    cq = df[df['source'] == 'code_query']

    # ── 1. Recall@k ─────────────────────────────────────────────────
    recall_df = cq.groupby(['embedder', 'graph_variant', 'k'])['recall'].mean().reset_index()
    recall_df['strategy'] = recall_df['embedder'] + '\n' + recall_df['graph_variant']
    sns.barplot(ax=ax_recall, data=recall_df, x='k', y='recall',
                hue='strategy', palette='husl', dodge=True)
    ax_recall.set_title('Recall@k  (code-query eval)', fontsize=13)
    ax_recall.set_xlabel('k')
    ax_recall.set_ylabel('Recall')
    ax_recall.set_ylim(0, 1)
    ax_recall.legend(title='Strategy', fontsize=7, title_fontsize=8)

    # ── 2. Precision@k ──────────────────────────────────────────────
    prec_df = cq.groupby(['embedder', 'graph_variant', 'k'])['precision'].mean().reset_index()
    prec_df['strategy'] = prec_df['embedder'] + '\n' + prec_df['graph_variant']
    sns.barplot(ax=ax_precision, data=prec_df, x='k', y='precision',
                hue='strategy', palette='husl', dodge=True)
    ax_precision.set_title('Precision@k  (code-query eval)', fontsize=13)
    ax_precision.set_xlabel('k')
    ax_precision.set_ylabel('Precision')
    ax_precision.set_ylim(0, 1)
    ax_precision.legend(title='Strategy', fontsize=7, title_fontsize=8)

    # ── 3. MRR by strategy ──────────────────────────────────────────
    mrr_df = (
        cq[cq['k'] == 10]
        .groupby(['embedder', 'graph_variant', 'backend'])['mrr']
        .mean()
        .reset_index()
    )
    mrr_df['strategy'] = mrr_df['embedder'] + ' / ' + mrr_df['graph_variant'] + ' / ' + mrr_df['backend']
    mrr_df = mrr_df.sort_values('mrr', ascending=True)
    ax_mrr.barh(mrr_df['strategy'], mrr_df['mrr'],
                color=[palette.get(e, 'steelblue') for e in mrr_df['embedder']])
    ax_mrr.set_title('Mean Reciprocal Rank (MRR) per Strategy', fontsize=13)
    ax_mrr.set_xlabel('MRR')
    ax_mrr.set_xlim(0, max(mrr_df['mrr'].max() * 1.2, 0.05))
    for i, (_, row) in enumerate(mrr_df.iterrows()):
        ax_mrr.text(row['mrr'] + 0.002, i, f"{row['mrr']:.3f}", va='center', fontsize=8)

    # ── 4. Summary table ────────────────────────────────────────────
    summary_rows = []
    for cell in raw_data['cells']:
        sr  = cell.get('self_retrieval', {})
        cwr = cell.get('cwe_recall', {})
        summary_rows.append({
            'Embedder':      cell['embedder'],
            'Graph':         cell['graph_variant'],
            'Backend':       cell['backend'],
            'Hit@1':         f"{sr.get('hit@1', 0):.3f}",
            'Hit@5':         f"{sr.get('hit@5', 0):.3f}",
            'Hit@10':        f"{sr.get('hit@10', 0):.3f}",
            'MRR':           f"{sr.get('mrr', 0):.3f}",
            'CWE Rec.':      f"{cwr.get('macro_avg', 0):.3f}",
            'n_cwes':        cwr.get('n_cwes', 0),
            'n_singletons':  cwr.get('n_singletons', 0),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(['Embedder', 'Graph', 'Backend'])

    ax_table.axis('off')
    col_labels = list(summary_df.columns)
    table_data = summary_df.values.tolist()
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))
    # header styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # alternate row shading
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    ax_table.set_title('Summary Table', fontsize=13, pad=10)

    return fig


def generate_visualizations(raw_data: dict, output_dir: str) -> None:
    """Generate and save both dashboards to output_dir."""
    import os
    output_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    df_perf = prepare_data(raw_data)
    create_dashboard(df_perf)
    plt.savefig(os.path.join(output_dir, 'dashboard_performance.png'), dpi=150, bbox_inches='tight')
    plt.close('all')

    df_qual = prepare_quality_data(raw_data)
    create_quality_dashboard(df_qual)
    plt.savefig(os.path.join(output_dir, 'dashboard_quality.png'), dpi=150, bbox_inches='tight')
    plt.close('all')

    df_metrics = prepare_metrics_data(raw_data)
    if not df_metrics.empty:
        create_metrics_dashboard(df_metrics, raw_data)
        plt.savefig(os.path.join(output_dir, 'dashboard_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close('all')

    print(f"Visualizations saved → {output_dir}")