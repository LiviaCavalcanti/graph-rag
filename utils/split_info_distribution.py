#!/usr/bin/env python3
"""Compute class (CVE / CWE) frequency distributions from a split_info.json file.

Usage:
    uv run python utils/split_info_distribution.py path/to/split_info.json
    uv run python utils/split_info_distribution.py path/to/split_info.json --out distribution.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def counter_summary(counter: Counter[str], total: int) -> list[dict[str, Any]]:
	return [
		{"key": key, "count": count, "frequency": count / total if total else 0.0}
		for key, count in counter.most_common()
	]


def compute_distribution(split_info: dict[str, Any]) -> dict[str, Any]:
	distribution: dict[str, Any] = {}

	for subset_name in ("index", "query"):
		entries = split_info.get(subset_name, [])
		total = len(entries)

		cve_counter = Counter(e["cve_id"] for e in entries)
		cwe_counter = Counter(e["cwe_id"] for e in entries)
		variant_counter = Counter(e.get("variant", "unknown") for e in entries)

		distribution[subset_name] = {
			"n_entries": total,
			"n_unique_cves": len(cve_counter),
			"n_unique_cwes": len(cwe_counter),
			"cve_distribution": counter_summary(cve_counter, total),
			"cwe_distribution": counter_summary(cwe_counter, total),
			"variant_distribution": counter_summary(variant_counter, total),
		}

	# Combined (index + query) stats
	all_entries = split_info.get("index", []) + split_info.get("query", [])
	total = len(all_entries)
	cve_counter = Counter(e["cve_id"] for e in all_entries)
	cwe_counter = Counter(e["cwe_id"] for e in all_entries)
	variant_counter = Counter(e.get("variant", "unknown") for e in all_entries)

	distribution["combined"] = {
		"n_entries": total,
		"n_unique_cves": len(cve_counter),
		"n_unique_cwes": len(cwe_counter),
		"cve_distribution": counter_summary(cve_counter, total),
		"cwe_distribution": counter_summary(cwe_counter, total),
		"variant_distribution": counter_summary(variant_counter, total),
	}

	return distribution


def print_summary(distribution: dict[str, Any]) -> None:
	for subset_name in ("index", "query", "combined"):
		subset = distribution[subset_name]
		print(f"\n=== {subset_name} (n={subset['n_entries']}) ===")
		print(f"unique CVEs: {subset['n_unique_cves']}, unique CWEs: {subset['n_unique_cwes']}")

		print("Top CWEs:")
		for row in subset["cwe_distribution"][:15]:
			print(f"  {row['key']:<15} count={row['count']:<4} freq={row['frequency']:.3f}")

		print("Top CVEs (should mostly be 1, duplicates indicate multiple functions/variants):")
		for row in subset["cve_distribution"][:10]:
			print(f"  {row['key']:<20} count={row['count']:<4} freq={row['frequency']:.3f}")

		print("Variant distribution:")
		for row in subset["variant_distribution"]:
			print(f"  {row['key']:<15} count={row['count']:<4} freq={row['frequency']:.3f}")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("split_info", type=Path, help="Path to split_info.json")
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Optional path to write the full distribution as JSON",
	)
	args = parser.parse_args()

	split_info = json.loads(args.split_info.read_text(encoding="utf-8"))
	distribution = compute_distribution(split_info)

	print_summary(distribution)

	if args.out:
		args.out.write_text(json.dumps(distribution, indent=2), encoding="utf-8")
		print(f"\nWrote full distribution to {args.out}")


if __name__ == "__main__":
	main()
