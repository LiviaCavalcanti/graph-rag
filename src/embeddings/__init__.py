from .motif import MotifEmbedder

from .netlsd import NetLSDEmbedder
from .wl import WLEmbedder
from .gin import GINEmbedder
from .combined import CombinedEmbedder
from .rgcn import RGCNEmbedder

REGISTRY: dict[str, type] = {
    "netlsd": NetLSDEmbedder,
    "wl": WLEmbedder,
    "gin": GINEmbedder,
    "combined": CombinedEmbedder,
    'motif':    MotifEmbedder,
    'rgcn':  RGCNEmbedder,
}


def build_embedders(cfg: dict) -> list:
    active = cfg["embeddings"]["active"]
    return [REGISTRY[name](cfg["embeddings"]) for name in active]
