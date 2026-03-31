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
        if self.config.get('include_variants', False):
            return _VARIANTS + [_ORIGINAL_VARIANT]
        else:
            return [_ORIGINAL_VARIANT]
    
    def _load_files(self):
        datset_path = self.config["path"]
        glob.glob(
        os.path.join(datset_path, "**", "export.xml"), recursive=True
    )

    def stream(self) -> Iterator[FunctionPair]:

        ...