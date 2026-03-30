from typing import Iterator

from .base import BaseDataset, FunctionPair

class AutoPatchDataset(BaseDataset):
    def name(self) -> str:
        return "autopatch"
    
    def stream(self) -> Iterator[FunctionPair]:

        ...