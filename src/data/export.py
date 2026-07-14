"""CPG export orchestration: run Joern over dataset export jobs in parallel.

Extracted from ``main.run_export`` — the CLI's ``--mode export`` now just
calls :func:`run_export`.
"""

from __future__ import annotations

from enum import Enum, auto
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from src.data import REGISTRY
from src.data.base import ExportJob
from src.data.pipeline import run_joern_export, write_c_file
from src.schema_config import AppConfig


class JobStatus(Enum):
    OK = auto()
    SKIPPED = auto()
    FAILED = auto()


def _process_job(job: ExportJob, joern_bin_dir: str) -> tuple[JobStatus, str]:
    out_dir = Path(job.out_dir)
    label = f"{job.cve_id}/{job.variant}/{job.version}"
    existing = list(out_dir.glob("**/export.xml"))
    if existing:
        return JobStatus.SKIPPED, f"skip {label} already exists"

    c_file = out_dir / f"{job.func_name or 'function'}.cpp"
    graph_folder = out_dir / "graph"

    try:
        write_c_file(job.source_code, c_file, supplementary_code=job.supplementary_code)
    except Exception as e:
        return JobStatus.FAILED, f"write failed {job.cve_id}: {e}"

    success = run_joern_export(joern_bin_dir, c_file, str(out_dir), str(graph_folder))
    if success:
        return JobStatus.OK, f"ok {label}"
    return JobStatus.FAILED, f"FAIL {label}"


def run_export(cfg: AppConfig, dataset_name: str | None = None, level: str = "method"):
    """Export CPGs for specified dataset(s).

    Args:
        cfg: AppConfig with paths and dataset configs
        dataset_name: Specific dataset to export ('autopatch', 'cvefixes', etc).
                     If None, uses all datasets in config.data.active
        level: 'method' (legacy per-function) or 'file' (one CPG per file)
    """
    joern_bin_dir = str(cfg.paths.joern_bin_dir)
    if not joern_bin_dir:
        raise KeyError("Joern path not found. Set paths.joern_bin_dir or joern.bin_dir")

    workers = max(1, cpu_count() - 1)

    # Determine which datasets to process
    if dataset_name:
        # Explicit dataset requested
        active = [dataset_name] if REGISTRY.get(dataset_name) else []
    else:
        # Use config's active datasets, defaulting to all if not specified
        if isinstance(cfg, AppConfig):
            active = cfg.raw.get("data", {}).get("active", list(REGISTRY.keys()))
        else:
            active = cfg.get("data", {}).get("active", list(REGISTRY.keys()))

    # Filter to configured datasets
    active = [n for n in active if cfg.data.get(n)]

    if not active:
        print("ERROR: No active datasets configured. Check config.yaml: data.active")
        return

    for ds_name in active:
        ds_cfg = cfg.data[ds_name]
        dataset = REGISTRY[ds_name](ds_cfg)
        graphml_root = ds_cfg["graphml_root"]

        print(
            f"\n -------------- exporting {dataset.name()} (level={level})--------------"
        )
        if level == "file" and hasattr(dataset, "export_jobs_file_level"):
            jobs = list(dataset.export_jobs_file_level(graphml_root))
        else:
            jobs = list(dataset.export_jobs(graphml_root))
        print(f" {len(jobs)} jobs & {workers} workers")

        worker_fn = partial(_process_job, joern_bin_dir=joern_bin_dir)

        ok = fail = skipped = 0
        with Pool(processes=workers) as pool:
            with tqdm(total=len(jobs), desc=ds_name, unit="job") as pbar:
                for status, msg in pool.imap_unordered(worker_fn, jobs, chunksize=4):
                    if status == JobStatus.SKIPPED:
                        skipped += 1
                    elif status == JobStatus.FAILED:
                        fail += 1
                        tqdm.write(f" {msg}")
                    else:
                        ok += 1
                    pbar.set_postfix(ok=ok, skip=skipped, fail=fail)
                    pbar.update(1)
        print(f"Done \n    ok: {ok}  -  skipped: {skipped}  -  fail: {fail}")
