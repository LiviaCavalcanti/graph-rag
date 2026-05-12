"""Targeted experiment: graph_v2 prompt on CVEs that failed with graph v1."""

from experiments.exp.patching_experiment import run_patching_experiment
from experiments.common import load_config

# CVEs that were NOT_FIXED under checkpoint5 graph variant
FAILED_CVES: set[str] = {
    "CVE-2022-49733",
    "CVE-2025-21998",
    "CVE-2025-21858",
    "CVE-2025-22004",
    "CVE-2025-21861",
    "CVE-2024-58022",
    "CVE-2024-58074",
    "CVE-2025-21901",
    "CVE-2024-58081",
    "CVE-2025-21809",
    "CVE-2025-21801",
    "CVE-2025-21807",
    "CVE-2025-21697",
    "CVE-2024-58082",
    "CVE-2024-36617",
    "CVE-2024-57254",
    "CVE-2025-21997",
    "CVE-2025-21799",
    "CVE-2025-21804",
    "CVE-2025-21838",
    "CVE-2025-21841",
    "CVE-2025-21842",
    "CVE-2025-21987",
}


def main():
    cfg = load_config()
    output_dir = run_patching_experiment(
        cfg,
        retriever_mode="oracle",
        prompt_variant="graph_v2",
        cve_filter=FAILED_CVES,
    )
    print(f"\n✓ Results written to: {output_dir}")


if __name__ == "__main__":
    main()
