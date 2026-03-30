from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class FunctionPair:
    """
    Canonical dataset loader.
    """
    cve_id: str
    func_name: str
    meta: dict = field(default_factory=dict)

class BaseDataset(ABC):

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def stream(self) -> Iterator[FunctionPair]:
        """
        yield a CVE function pair
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """
        Identifier for logging and metadata
        """

    def load_all(self):
        return list(self.stream())