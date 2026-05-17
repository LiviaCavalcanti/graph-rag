# experiments/visualize_diagnostics.py
# uv run python experiments/visualize_diagnostics.py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load(path='experiments/diagnostics/diagnostics.json'):
    return json.loads(Path(path).read_text())


def plot_all(data: dict, out_dir='experiments/diagnostics'):
    out = Path(out_dir)

    # ── 1. node count distribution per CWE ───────────────────────────
    # tells you directly which CWEs have too-small G_vuln
    cwe_nodes = defaultdict(list)
    for r in data['per_graph']:
        if r['variant'] == 'G_vuln':
            cwe_nodes[r['cwe_id']].append(r['n_nodes'])

    fig, ax = plt.subplots(figsize=(12, 5))
    labels  = sorted(cwe_nodes.keys(), key=lambda c: np.median(cwe_nodes[c]))
    ax.boxplot([cwe_nodes[c] for c in labels], vert=True, patch_artist=True)
    ax.axhline(y=5, color='red', linestyle='--', label='min useful size (5)')
    ax.axhline(y=10, color='orange', linestyle='--', label='good size (10)')
    ax.set_xticklabels([c[:25] for c in labels], rotation=45, ha='right')
    ax.set_ylabel('G_vuln node count')
    ax.set_title('G_vuln size distribution per CWE\n'
                 '(red=too small for NetLSD, orange=minimum useful)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / 'vuln_size_by_cwe.png', dpi=150)
    plt.close()

    # ── 2. G_before vs G_vuln size ratio ─────────────────────────────
    # if ratio is consistently <0.1, diff is too aggressive
    before = {(i, r['cve_id']): r['n_nodes']
               for i, r in enumerate(data['per_graph']) if r['variant'] == 'G_before'}
    vuln   = {(i, r['cve_id']): r['n_nodes']
               for i, r in enumerate(data['per_graph']) if r['variant'] == 'G_vuln'}
    # pair up by position: every G_before record at index i corresponds to G_vuln at the same position
    before_vals = [r['n_nodes'] for r in data['per_graph'] if r['variant'] == 'G_before']
    vuln_vals   = [r['n_nodes'] for r in data['per_graph'] if r['variant'] == 'G_vuln']
    ratios = [v / max(b, 1) for v, b in zip(vuln_vals, before_vals)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ratios, bins=30, edgecolor='black')
    ax.axvline(x=0.1, color='red', linestyle='--', label='<10% of G_before')
    ax.set_xlabel('G_vuln nodes / G_before nodes')
    ax.set_ylabel('count')
    ax.set_title('Diff subgraph size relative to full CPG\n'
                 '(left-heavy = diff too aggressive)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / 'diff_ratio.png', dpi=150)
    plt.close()

    # ── 3. edge type heatmap per CWE ─────────────────────────────────
    # reveals which CWEs have CFG/CDG edges (structural signal)
    # vs only AST edges (tree-like, low discriminability)
    edge_types = ['AST', 'CFG', 'CDG', 'REF', 'ARGUMENT',
                  'RECEIVER', 'REACHING_DEF']
    cwes       = sorted(data['cwe_summary'].keys())
    matrix     = np.zeros((len(cwes), len(edge_types)))

    for i, cwe in enumerate(cwes):
        etypes = data['cwe_summary'][cwe].get('edge_types', {})
        total  = sum(etypes.values()) or 1
        for j, et in enumerate(edge_types):
            matrix[i, j] = etypes.get(et, 0) / total

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(edge_types)))
    ax.set_xticklabels(edge_types, rotation=45, ha='right')
    ax.set_yticks(range(len(cwes)))
    ax.set_yticklabels([c[:40] for c in cwes])
    ax.set_title('Edge type distribution per CWE in G_vuln\n'
                 '(AST-dominant = low structural discriminability)')
    plt.colorbar(im, ax=ax, label='fraction of edges')
    plt.tight_layout()
    plt.savefig(out / 'edge_type_heatmap.png', dpi=150)
    plt.close()

    # ── 4. intra vs inter CWE similarity ─────────────────────────────
    # if intra ≈ inter, embedder cannot separate CWE clusters
    intra = data['cwe_intra_sim']
    cwe_labels  = list(intra.keys())
    intra_means = [intra[c]['mean_intra_sim'] for c in cwe_labels]
    intra_stds  = [intra[c]['std_intra_sim']  for c in cwe_labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cwe_labels))
    ax.bar(x, intra_means, yerr=intra_stds, capsize=4)
    ax.axhline(y=np.mean(intra_means), color='red',
               linestyle='--', label=f'macro avg={np.mean(intra_means):.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels([c[:25] for c in cwe_labels], rotation=45, ha='right')
    ax.set_ylabel('mean intra-CWE cosine similarity')
    ax.set_title('Structural similarity within each CWE group\n'
                 '(high = graphs in this CWE look similar to each other)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / 'intra_cwe_similarity.png', dpi=150)
    plt.close()

    # ── 5. CODE attribute coverage ────────────────────────────────────
    # if code_coverage < 0.5, semantic embedders won't help much
    cov_by_cwe = defaultdict(list)
    for r in data['per_graph']:
        if r['variant'] == 'G_vuln':
            cov_by_cwe[r['cwe_id']].append(r['code_coverage'])

    fig, ax = plt.subplots(figsize=(10, 4))
    labels = sorted(cov_by_cwe.keys())
    ax.boxplot([cov_by_cwe[c] for c in labels])
    ax.set_xticklabels([c[:25] for c in labels], rotation=45, ha='right')
    ax.set_ylabel('fraction of nodes with CODE attribute')
    ax.set_title('CODE attribute coverage per CWE\n'
                 '(low = semantic embeddings will not help)')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='50% threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / 'code_coverage.png', dpi=150)
    plt.close()

    # ── 6. t-SNE of node-type histograms, coloured by CWE ────────────
    # visual check: are any CWEs separable in embedding space at all?
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import LabelEncoder
        import matplotlib.cm as cm

        types = ['METHOD','CALL','IDENTIFIER','LITERAL','RETURN',
                 'BLOCK','CONTROL_STRUCTURE','LOCAL','PARAM','UNKNOWN']
        rows, cwe_ids = [], []
        for r in data['per_graph']:
            if r['variant'] == 'G_vuln' and r['n_nodes'] > 3:
                nt    = r['node_types']
                total = max(sum(nt.values()), 1)
                rows.append([nt.get(t, 0) / total for t in types])
                cwe_ids.append(r['cwe_id'])

        if len(rows) > 10:
            X   = np.array(rows)
            le  = LabelEncoder()
            y   = le.fit_transform(cwe_ids)
            emb = TSNE(n_components=2, random_state=42,
                       perplexity=min(30, len(rows)//3)).fit_transform(X)

            fig, ax = plt.subplots(figsize=(9, 7))
            colors  = cm.tab20(np.linspace(0, 1, len(le.classes_)))
            for i, cwe in enumerate(le.classes_):
                mask = y == i
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=[colors[i]], label=cwe[:30], s=40, alpha=0.7)
            ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1))
            ax.set_title('t-SNE of node-type histograms coloured by CWE\n'
                         '(clusters = CWEs are structurally separable)')
            plt.tight_layout()
            plt.savefig(out / 'tsne_node_types.png', dpi=150, bbox_inches='tight')
            plt.close()
    except ImportError:
        print("  sklearn not available — skipping t-SNE")

    print(f"\nPlots saved → {out}/")


if __name__ == '__main__':
    plot_all(load())