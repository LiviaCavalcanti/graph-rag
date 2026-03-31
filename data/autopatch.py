from typing import Iterator
import glob
import os
from .base import BaseDataset, FunctionPair

class AutoPatchDataset(BaseDataset):
    def name(self) -> str:
        return "AutoPatch"
    
    def _load_files(self):
        datset_path = self.config["path"]
        glob.glob(
        os.path.join(datset_path, "**", "export.xml"), recursive=True
    )

    def stream(self) -> Iterator[FunctionPair]:

        ...