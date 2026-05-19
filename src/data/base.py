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
    supplementary_code: str = ""  # to support autopatch


@dataclass
class FunctionPair:
    """
    Canonical dataset loader.
    """

    cve_id: str
    cwe_id: str
    func_name: str
    G_before: nx.MultiDiGraph
    G_after: nx.MultiDiGraph
    G_vuln: nx.MultiDiGraph
    project: str
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

    def load_lightweight(self) -> list["FunctionPair"]:
        """Load pairs with metadata only — no CPG/graph loading.

        Default falls back to load_all(). Override for faster
        metadata-only loading when graphs aren't needed.
        """
        return self.load_all()

    @abstractmethod
    def export_jobs(self, graphml_root: str) -> Iterator[ExportJob]: ...
