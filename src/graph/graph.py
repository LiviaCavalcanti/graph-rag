import glob
import json
import os
import subprocess

import networkx as nx


class CPGGenerator:
    def __init__(self, joern_path):
        self.joern_path = joern_path

    def generate_cpg(self, source_code_path, output_bin):
        """Runs joern-parse to create a binary CPG file."""
        cmd = [
            os.path.join(self.joern_path, "joern-parse"),
            source_code_path,
            "--output",
            output_bin,
        ]
        subprocess.run(cmd, check=True)
        print(f"CPG binary generated at: {output_bin}")

    def export_to_graphml(self, cpg_bin, output_path):
        """Uses joern-export to create a GraphML file."""
        export_script = os.path.join(self.joern_path, "joern-export")

        # Joern-export creates a DIRECTORY out
        cmd = [
            export_script,
            "--repr",
            "cpg",
            "--format",
            "graphml",
            cpg_bin,
            "--out",
            output_path,
        ]

        # Clean up existing directory to avoid "File exists" errors
        if os.path.exists(output_path):
            import shutil

            shutil.rmtree(output_path)

        subprocess.run(cmd, check=True)
        print(f"CPG exported to GraphML folder: {output_path}")


def manipulate_graph(graphml_dir):
    # Find all the Joern exported each function in every directory
    # each generated file is named export.xml/
    export_files = glob.glob(
        os.path.join(graphml_dir, "**", "export.xml"), recursive=True
    )

    if not export_files:
        raise FileNotFoundError(f"No Joern export files found under: {graphml_dir}")

    failed = []
    G = nx.MultiDiGraph()
    for f in export_files:
        try:
            sub_graph = nx.read_graphml(f, node_type=str, force_multigraph=True)
            G.update(sub_graph)
        except Exception as e:
            failed.append((f, str(e)))

    if failed:
        print(f"Warning: {len(failed)} files failed to parse:")
        for path, err in failed:
            print(f"  {path}: {err}")

    nodes_to_remove = [
        n
        for n, attr in G.nodes(data=True)
        if attr.get("labelV") == "COMMENT" or attr.get("nodeType") == "COMMENT"
    ]
    G.remove_nodes_from(nodes_to_remove)

    print(f"Loaded {len(export_files) - len(failed)}/{len(export_files)} graphs")
    print(f"Total — Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    return G

    import json


def export_graph_json(G, out_path="graph_viz.json"):
    """
    One file with the whole graph for visualization
    """
    nodes = []
    for n, attr in G.nodes(data=True):
        nodes.append(
            {
                "id": str(n),
                "label": attr.get("CODE", attr.get("NAME", attr.get("labelV", str(n))))[
                    :40
                ],
                "type": attr.get("labelV", attr.get("nodeType", "UNKNOWN")),
            }
        )

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append(
            {
                "src": str(u),
                "dst": str(v),
                "label": data.get("label", data.get("labelE", "EDGE")),
            }
        )

    with open(out_path, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f)

    print(f"Exported {len(nodes)} nodes, {len(edges)} edges → {out_path}")


# --- Execution Workflow ---
# CHANGE THIS: Use the WSL mount path and forward slashes
JOERN_BIN_DIR = "/home/z0050s2b/bin/joern/joern-cli"
source_dir = "test_input.c"
cpg_file = "entry_1.bin"
graphml_out = "entry_1_graphml"

generator = CPGGenerator(JOERN_BIN_DIR)

# 1. Create the CPG
generator.generate_cpg(source_dir, cpg_file)

# 2. Export it so Python can read it
generator.export_to_graphml(cpg_file, graphml_out)

# 3. Manipulate via NetworkX
# Note: joern-export creates a directory of files; you'd point to the specific export
graph = manipulate_graph(f"{graphml_out}")

export_graph_json(graph, "graph_viz.json")

# 4. Save to a simple output file (GraphML is great for visualization tools like Gephi)
nx.write_graphml(graph, "final_cpg.graphml")
