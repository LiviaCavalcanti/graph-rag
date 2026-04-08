from .netlsd import NetLSDEmbedder
from .wl import WLEmbedder
from .gin import GINEmbedder
from .combined import CombinedEmbedder

REGISTRY: dict[str, type] = {
    "netlsd": NetLSDEmbedder,
    "wl": WLEmbedder,
    "gin": GINEmbedder,
    "combined": CombinedEmbedder,
}


def build_embedders(cfg: dict) -> list:
    active = cfg["embeddings"]["active"]
    return [REGISTRY[name](cfg["embeddings"]) for name in active]
