import json
from pathlib import Path
from typing import Iterator
from .base import BaseDataset, FunctionPair
from .pipeline import load_function_graph, compute_graph_diff

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
        self, cve_id: str, func_name: str, variant: str, meta: dict
    ) -> FunctionPair | None:
        G_before = load_function_graph(func_name, cve_id, hint=variant)

        G_after = load_function_graph(func_name, cve_id, hint=variant)

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
        files_to_use = self._variants_to_use()

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
                if pair is not None:
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
