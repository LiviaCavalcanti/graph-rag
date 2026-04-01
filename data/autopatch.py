import json
from pathlib import Path
from typing import Iterator

from graph.joern_graph import get_cpg

from .base import BaseDataset, FunctionPair, ExportJob
from .pipeline import compute_graph_diff, load_function_graph

# augmented versions
_VARIANTS = [
    ("augmented.json", "augmented_fixed.c"),
    ("patch_augmented.json", "patch_augmented_fixed.c"),
    ("re_implemented_deepseek.json", "re_implemented_deepseek_fixed.c"),
    ("re_implemented_deepseek-r1.json", "re_implemented_deepseek-r1_fixed.c"),
    ("re_implemented_gpt-4o.json", "re_implemented_gpt-4o_fixed.c"),
    ("re_implemented_llama.json", "re_implemented_llama_fixed.c"),
    ("re_implemented_o3-mini.json", "re_implemented_o3-mini_fixed.c"),
]


class AutoPatchDataset(BaseDataset):
    """
    Each CVE folder has this structure:
        CVE-list/CVE-XXXX-YYYY/
            original_code.txt       ← vulnerable function (ground truth)
            original_code_fixed.c   ← patched function (ground truth)
            supplementary_code.txt
            vuln_patch.txt
            out_v2/code/            ← augmented + model variants (optional)
            out_v2/db_entry.json    ← CVE metadata

    """

    def name(self) -> str:
        return "AutoPatch"

    def _load_variant_json(self, code_dir: Path, json_file: str) -> dict | None:
        path = code_dir / json_file
        if not path.exists():
            return None

        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def _make_pair(
        self, root: str, cve_id: str, func_name: str, variant: str, meta: dict
    ) -> FunctionPair | None:
        G_before = load_function_graph(
            root, version="before", func_name=func_name, cve_id=cve_id, hint=variant
        )
        G_after = load_function_graph(
            root, version="after", func_name=func_name, cve_id=cve_id, hint=variant
        )

        if G_before is None and G_after is None:
            return None
        if G_before.number_of_nodes() == 0 or G_after.number_of_nodes() == 0:
            return None

        return FunctionPair(
            cve_id=cve_id,
            func_name=func_name,
            dataset=self.name(),
            G_before=G_before,
            G_after=G_after,
            G_vuln=compute_graph_diff(G_before, G_after),
            meta=meta,
        )

    def _load_db_entry(self, cve_dir: Path):
        db_path = cve_dir / "out_v2" / "db_entry.json"
        if not db_path.exists():
            return None
        try:
            return json.loads(db_path.read_text())
        except json.JSONDecoderError:
            return None

    def stream(self) -> Iterator[FunctionPair]:
        dataset_path = Path(self.cfg["path"])
        root = self.cfg["graphml_root"]

        for cve_dir in sorted(dataset_path.iterdir()):
            print(cve_dir)
            db = self._load_db_entry(cve_dir)
            if db is None:
                continue
            cve_id = str(db.get("cve_id", cve_dir.name))
            # cwe_id    = str(db.get('cwe_type', ''))
            func_name = str(db.get("function_name", ""))

            base_meta = {
                "root_cause": db.get("root_cause", ""),
                "fix_list": db.get("fix_list", []),
                "function_prototype": db.get("function_prototype", ""),
            }

            # original code
            original_code_path = cve_dir / "original_code.txt"
            original_fixed_path = cve_dir / "original_code_fixed.c"

            if original_code_path.exists() and original_fixed_path.exists():
                pair = self._make_pair(
                    root,
                    cve_id,
                    func_name,
                    variant="original",
                    meta={
                        "dataset": self.name(),
                        "variant": "original",
                        "source_before": str(original_code_path),
                        "source_after": str(original_fixed_path),
                        "supplementary_code": (
                            (cve_dir / "supplementary_code.txt").read_text()
                            if (cve_dir / "supplementary_code.txt").exists()
                            else ""
                        ),
                        **base_meta,
                    },
                )
                print(f"{pair}\n ------------------------\n\n\n")
                if pair is not None:
                    print(f"{pair}\n ------------------------")
                    yield pair

            # augmented data
            if self.cfg.get("include_variants", False):
                code_dir = cve_dir / "out_v2" / "code"
                if not code_dir.exists():
                    for json_file, fixed_file in _VARIANTS:
                        variant_data = self._load_variant_json(code_dir, json_file)
                        if variant_data:
                            # only consider the entries that have a vulnerability
                            if not variant_data.get("is_vulnerable", False):
                                continue
                            fixed_c_path = code_dir / fixed_file
                            if fixed_c_path.exists():
                                variant_name = json_file.replace(".json", "")

                                pair = self._make_pair(
                                    root,
                                    cve_id,
                                    func_name,
                                    meta={
                                        "dataset": self.name(),
                                        "variant": variant_name,
                                        "source_before": variant_data.get(
                                            "re_implemented_code", ""
                                        ),
                                        "source_after": str(fixed_c_path),
                                        "supplementary_code": variant_data.get(
                                            "supplementary_code", ""
                                        ),
                                        **base_meta,
                                    },
                                )

                                yield pair

    def export_jobs(self, graphml_root: str) -> Iterator[ExportJob]:
        root = Path(self.cfg["root"])
        variants = self._variants_to_use()

        for cve_dir in sorted(root.iterdir()):
            if not cve_dir.is_dir():
                continue
            db = self._load_db_entry(cve_dir=cve_dir)
            if db is None:
                continue

            cve_id = str(db.get("cve_id", cve_dir.name))
            func_name = str(db.get("function_name", ""))
            base = Path(graphml_root) / cve_id

            before_path = cve_dir / "original_code.txt"
            after_path = cve_dir / "original_code_fixed.txt"

            if before_path.exists() and after_path.exists():
                yield ExportJob(
                    cve_id=cve_id,
                    func_name=func_name,
                    variant="original",
                    source_code=before_path.read_text(),
                    version="before",
                    out_dir=str(base / "original" / "before"),
                )

                yield ExportJob(
                    cve_id=cve_id,
                    func_name=func_name,
                    variant="original",
                    source_code=after_path.read_text(),
                    version="after",
                    out_dir=str(base / "original" / "after"),
                )

                if self.cfg.get("include_variants", False):
                    code_dir = cve_dir / "out_v2" / "code"
                    if code_dir.exists():
                        for json_file, fixed_c_file in variants:
                            variant_data = self._load_variant_json(
                                code_dir=code_dir, json_file=json_file
                            )

                            if variant_data is not None and not variant_data.get(
                                "is_vulnerable", False
                            ):
                                fixed_c_path = code_dir / fixed_c_file

                                if fixed_c_file.exists():
                                    variant_name = json_file.replace(".json", "")

                                    yield ExportJob(
                                        cve_id=cve_id,
                                        func_name=func_name,
                                        variant="augmented",
                                        version="before",
                                        source_code=variant_data.get(
                                            "re_implemented_code", ""
                                        ),
                                        out_dir=str(base / variant_name / "before"),
                                    )
                                    yield ExportJob(
                                        cve_id=cve_id,
                                        func_name=func_name,
                                        variant="augmented",
                                        version="after",
                                        source_code=fixed_c_path.read_text(),
                                        out_dir=str(base / variant_name / "after"),
                                    )
