from .motif import MotifEmbedder

from .netlsd import NetLSDEmbedder
from .wl import WLEmbedder
from .gin import GINEmbedder
from .combined import CombinedEmbedder
from .rgcn import RGCNEmbedder
from .codebert_seq import CodeBERTSeqEmbedder
from .vuln_pattern import VulnPatternEmbedder, CodeBERTPatternEmbedder
from .codexglue_baseline import CodeXGLUEBaselineEmbedder

REGISTRY: dict[str, type] = {
    "netlsd": NetLSDEmbedder,
    "wl": WLEmbedder,
    "gin": GINEmbedder,
    "combined": CombinedEmbedder,
    'motif':    MotifEmbedder,
    'rgcn':  RGCNEmbedder,
    'codebert_seq':    CodeBERTSeqEmbedder,
    'vuln_pattern':    VulnPatternEmbedder,
    'codebert_pattern': CodeBERTPatternEmbedder,
    'codexglue_baseline': CodeXGLUEBaselineEmbedder,
}


def build_embedders(cfg: dict) -> list:
    active = cfg["embeddings"]["active"]
    return [REGISTRY[name](cfg["embeddings"]) for name in active]
