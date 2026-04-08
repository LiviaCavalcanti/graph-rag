# experiments/diagnostics.py
import json
import os
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from pathlib import Path


def run_diagnostics(pairs: list, output_dir: str = 'experiments/diagnostics'):
    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    cwe_node_counts   = defaultdict(list)
    cwe_edge_types    = defaultdict(Counter)
    node_type_global  = Counter()
    edge_type_global  = Counter()

    for p in pairs:
        for variant, G in [('G_vuln', p.G_vuln), ('G_before', p.G_before)]:
            n_nodes  = G.number_of_nodes()
            n_edges  = G.number_of_edges()
            ntypes   = Counter(attr.get('labelV')
                               for _, attr in G.nodes(data=True))
            etypes   = Counter(d.get('labelE') or d.get('label')
                               for _, _, d in G.edges(data=True))
            has_code = sum(1 for _, a in G.nodes(data=True) if a.get('CODE','').strip())

            rec = {
                'cve_id':         p.cve_id,
                'cwe_id':         p.cwe_id,
                'variant':        variant,
                'n_nodes':        n_nodes,
                'n_edges':        n_edges,
                'n_edge_types':   len(etypes),
                'has_code_attr':  has_code,
                'code_coverage':  has_code / max(n_nodes, 1),
                'node_types':     dict(ntypes),
                'edge_types':     dict(etypes),
                'is_connected':   nx.is_weakly_connected(G) if n_nodes > 0 else False,
                'n_components':   nx.number_weakly_connected_components(G) if n_nodes > 0 else 0,
                # fraction of nodes that are 'interesting' (not scaffold artifacts)
                'pct_call_nodes': ntypes.get('CALL', 0) / max(n_nodes, 1),
                'pct_cfg_edges':  etypes.get('CFG', 0) / max(n_edges, 1),
                'pct_cdg_edges':  etypes.get('CDG', 0) / max(n_edges, 1),
                'pct_pdg_edges':  (etypes.get('REACHING_DEF', 0) +
                                   etypes.get('CDG', 0)) / max(n_edges, 1),
            }
            records.append(rec)

            if variant == 'G_vuln':
                cwe_node_counts[p.cwe_id].append(n_nodes)
                cwe_edge_types[p.cwe_id].update(etypes)
                node_type_global.update(ntypes)
                edge_type_global.update(etypes)

    # ── per-CWE summary ───────────────────────────────────────────────
    cwe_summary = {}
    for cwe, counts in cwe_node_counts.items():
        cwe_summary[cwe] = {
            'n_samples':      len(counts),
            'median_nodes':   float(np.median(counts)),
            'mean_nodes':     float(np.mean(counts)),
            'min_nodes':      int(np.min(counts)),
            'max_nodes':      int(np.max(counts)),
            'pct_tiny':       sum(1 for c in counts if c < 5) / len(counts),
            'pct_empty':      sum(1 for c in counts if c == 0) / len(counts),
            'edge_types':     dict(cwe_edge_types[cwe]),
        }

    # ── graph similarity matrix per CWE ──────────────────────────────
    # are graphs within a CWE more similar to each other than to others?
    # use node-type histogram cosine sim as a cheap proxy
    def node_hist(G):
        types  = ['METHOD','CALL','IDENTIFIER','LITERAL','RETURN',
                  'BLOCK','CONTROL_STRUCTURE','LOCAL','PARAM']
        counts = Counter(a.get('labelV','UNKNOWN') for _,a in G.nodes(data=True))
        vec    = np.array([counts.get(t, 0) for t in types], dtype=float)
        norm   = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    cwe_intra_sim = {}
    for cwe in set(p.cwe_id for p in pairs):
        cwe_pairs = [p for p in pairs if p.cwe_id == cwe]
        if len(cwe_pairs) < 2:
            continue
        vecs  = np.stack([node_hist(p.G_vuln) for p in cwe_pairs])
        sims  = vecs @ vecs.T
        mask  = ~np.eye(len(vecs), dtype=bool)
        cwe_intra_sim[cwe] = {
            'mean_intra_sim': float(sims[mask].mean()),
            'std_intra_sim':  float(sims[mask].std()),
        }

    output = {
        'per_graph':      records,
        'cwe_summary':    cwe_summary,
        'cwe_intra_sim':  cwe_intra_sim,
        'global_node_types': dict(node_type_global),
        'global_edge_types': dict(edge_type_global),
    }

    path = out / 'diagnostics.json'
    path.write_text(json.dumps(output, indent=2))
    print(f"Diagnostics written → {path}")

    # ── print actionable summary ──────────────────────────────────────
    print("\n── CWE node count summary (G_vuln) ──")
    for cwe, s in sorted(cwe_summary.items(), key=lambda x: x[1]['median_nodes']):
        print(f"  {cwe[:45]:<45} "
              f"median={s['median_nodes']:.0f}  "
              f"empty={s['pct_empty']:.0%}  "
              f"tiny={s['pct_tiny']:.0%}")

    print("\n── global edge type distribution ──")
    total_edges = sum(edge_type_global.values())
    for etype, count in edge_type_global.most_common():
        print(f"  {etype:<20} {count/total_edges:.1%}")

    print("\n── intra-CWE structural similarity ──")
    for cwe, s in sorted(cwe_intra_sim.items(),
                         key=lambda x: -x[1]['mean_intra_sim']):
        print(f"  {cwe[:45]:<45} "
              f"mean_sim={s['mean_intra_sim']:.3f}  "
              f"std={s['std_intra_sim']:.3f}")

    return output
