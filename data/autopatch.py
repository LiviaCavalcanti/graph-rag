import json
from pathlib import Path
from typing import Iterator
import glob
import os
from .base import BaseDataset, FunctionPair

# augmented versions
_VARIANTS = [
    ('augmented.json',                  'augmented_fixed.c'),
    ('patch_augmented.json',            'patch_augmented_fixed.c'),
    ('re_implemented_deepseek.json',    're_implemented_deepseek_fixed.c'),
    ('re_implemented_deepseek-r1.json', 're_implemented_deepseek-r1_fixed.c'),
    ('re_implemented_gpt-4o.json',      're_implemented_gpt-4o_fixed.c'),
    ('re_implemented_llama.json',       're_implemented_llama_fixed.c'),
    ('re_implemented_o3-mini.json',     're_implemented_o3-mini_fixed.c'),
]
# original cve example
_ORIGINAL_VARIANT = ('original_code.txt', 'original_code_fixed.c')

class AutoPatchDataset(BaseDataset):
    def name(self) -> str:
        return "AutoPatch"
    
    def _variants_to_use(self) -> list[tuple[str, str]]:
        if self.cfg.get('include_variants', False):
            return _VARIANTS + [_ORIGINAL_VARIANT]
        else:
            return [_ORIGINAL_VARIANT]
    
    def _load_db_entry(self, cve_dir: Path):
        db_path = cve_dir / 'out_v2' / 'db_entry.json'
        if not db_path.exists():
            return None
        try:
            return json.loads(db_path.read_text())
        except json.JSONDecoderError:
            return None
     

    def stream(self) -> Iterator[FunctionPair]:
        dataset_path = Path(self.cfg["path"])
        print()
        files_to_use = self._variants_to_use()

        for cve_dir in sorted(dataset_path.iterdir()):
            print(cve_dir)
            db = self._load_db_entry(cve_dir)
            cve_id    = str(db.get('cve_id', cve_dir.name))
            cwe_id    = str(db.get('cwe_type', ''))
            func_name = str(db.get('function_name', ''))

            yield FunctionPair(
                cve_id    = cve_id,
                func_name = func_name,
            )
        