from .codebert_seq import CodeBERTSeqEmbedder
from .codexglue_baseline import CodeXGLUEBaselineEmbedder
from .combined import CombinedEmbedder
from .gin import GINEmbedder
from .motif import MotifEmbedder
from .netlsd import NetLSDEmbedder
from .rgcn import RGCNEmbedder
from .vuln_pattern import CodeBERTPatternEmbedder, VulnPatternEmbedder
from .wl import WLEmbedder

REGISTRY: dict[str, type] = {
    "netlsd": NetLSDEmbedder,
    "wl": WLEmbedder,
    "gin": GINEmbedder,
    "combined": CombinedEmbedder,
    "motif": MotifEmbedder,
    "rgcn": RGCNEmbedder,
    "codebert_seq": CodeBERTSeqEmbedder,
    "vuln_pattern": VulnPatternEmbedder,
    "codebert_pattern": CodeBERTPatternEmbedder,
    "codexglue_baseline": CodeXGLUEBaselineEmbedder,
}


def build_embedders(cfg: dict) -> list:
    active = cfg["embeddings"]["active"]
    return [REGISTRY[name](cfg["embeddings"]) for name in active]
