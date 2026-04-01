from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator
import networkx as nx


@dataclass
class ExportJob:
    cve_id: str
    func_name: str
    variant: str  # original, augmented (autopatch)
    version: str  # before(vulnerable), after(patched)
    source_code: str
    out_dir: str  # final detination for export.xml


@dataclass
class FunctionPair:
    """
    Canonical dataset loader.
    """

    cve_id: str
    func_name: str
    meta: dict = field(default_factory=dict)
    G_before: nx.MultiDiGraph
    G_after: nx.MultiDiGraph
    G_vuln: nx.MultiDiGraph
    project: str


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
