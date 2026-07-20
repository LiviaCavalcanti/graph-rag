#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class FileCveData:
	path: Path
	unique_cves: set[str]
	total_cve_entries: int
	invalid_json_lines: int
	missing_query_cve_lines: int


def read_query_cves(jsonl_path: Path) -> FileCveData:
	unique_cves: set[str] = set()
	total_cve_entries = 0
	invalid_json_lines = 0
	missing_query_cve_lines = 0

	with jsonl_path.open("r", encoding="utf-8") as f:
		for line_number, raw_line in enumerate(f, start=1):
			line = raw_line.strip()
			if not line:
				continue

			try:
				row = json.loads(line)
			except json.JSONDecodeError:
				invalid_json_lines += 1
				continue

			if not isinstance(row, dict):
				missing_query_cve_lines += 1
				continue

			query_cve = row.get("query_cve")
			if isinstance(query_cve, str) and query_cve:
				unique_cves.add(query_cve)
				total_cve_entries += 1
			else:
				missing_query_cve_lines += 1

	return FileCveData(
		path=jsonl_path,
		unique_cves=unique_cves,
		total_cve_entries=total_cve_entries,
		invalid_json_lines=invalid_json_lines,
		missing_query_cve_lines=missing_query_cve_lines,
	)


def pairwise(items: list[FileCveData]) -> Iterable[tuple[FileCveData, FileCveData]]:
	for i in range(len(items)):
		for j in range(i + 1, len(items)):
			yield items[i], items[j]


def print_file_summary(data: FileCveData) -> None:
	print(f"File: {data.path}")
	print(f"  Total query_cve entries: {data.total_cve_entries}")
	print(f"  Unique query_cve values: {len(data.unique_cves)}")
	print(f"  Invalid JSON lines: {data.invalid_json_lines}")
	print(f"  Missing/non-string query_cve lines: {data.missing_query_cve_lines}")


def print_pairwise_comparison(left: FileCveData, right: FileCveData, show_values: bool) -> None:
	overlap = left.unique_cves & right.unique_cves
	left_only = left.unique_cves - right.unique_cves
	right_only = right.unique_cves - left.unique_cves

	print("\nComparison")
	print(f"  A: {left.path}")
	print(f"  B: {right.path}")
	print(f"  Overlap count: {len(overlap)}")
	print(f"  Only in A: {len(left_only)}")
	print(f"  Only in B: {len(right_only)}")
	print(f"  Same unique CVE set: {not left_only and not right_only}")

	if show_values:
		if left_only:
			print("  Values only in A:")
			for cve in sorted(left_only):
				print(f"    - {cve}")
		if right_only:
			print("  Values only in B:")
			for cve in sorted(right_only):
				print(f"    - {cve}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compare overlap of query_cve values across one or more JSONL files."
	)
	parser.add_argument(
		"files",
		nargs="+",
		help="Paths to JSONL files that contain query_cve fields.",
	)
	parser.add_argument(
		"--show-values",
		action="store_true",
		help="Print CVE values that are present only in one file during comparisons.",
	)
	args = parser.parse_args()

	file_paths = [Path(p) for p in args.files]
	missing = [p for p in file_paths if not p.exists()]
	if missing:
		raise SystemExit("Missing file(s):\n" + "\n".join(str(p) for p in missing))

	parsed = [read_query_cves(path) for path in file_paths]

	print("=== Query CVE Summary ===")
	for item in parsed:
		print_file_summary(item)

	if len(parsed) == 1:
		return

	print("\n=== Pairwise Overlap ===")
	for left, right in pairwise(parsed):
		print_pairwise_comparison(left, right, show_values=args.show_values)


if __name__ == "__main__":
	main()
