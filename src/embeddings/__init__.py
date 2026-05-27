from .codebert_seq import CodeBERTFlowEmbedder, CodeBERTSeqEmbedder
from .codexglue_baseline import CodeXGLUEBaselineEmbedder
from .combined import CombinedEmbedder, CombinedEnrichedEmbedder
from .gin import GINEmbedder, GINEnrichedEmbedder
from .motif import MotifEmbedder
from .netlsd import NetLSDEmbedder
from .rgcn import RGCNEmbedder
from .vuln_pattern import CodeBERTFlowPatternEmbedder, CodeBERTPatternEmbedder, VulnPatternEmbedder
from .wl import WLEmbedder

REGISTRY: dict[str, type] = {
    "netlsd": NetLSDEmbedder,
    "wl": WLEmbedder,
    "gin": GINEmbedder,
    "gin_enriched": GINEnrichedEmbedder,
    "combined": CombinedEmbedder,
    "combined_enriched": CombinedEnrichedEmbedder,
    "motif": MotifEmbedder,
    "rgcn": RGCNEmbedder,
    "codebert_seq": CodeBERTSeqEmbedder,
    "vuln_pattern": VulnPatternEmbedder,
    "codebert_pattern": CodeBERTPatternEmbedder,
    "codebert_flow": CodeBERTFlowEmbedder,
    "codebert_flow_pattern": CodeBERTFlowPatternEmbedder,
    "codexglue_baseline": CodeXGLUEBaselineEmbedder,
}


def build_embedders(cfg: dict) -> list:
    active = cfg["embeddings"]["active"]
    return [REGISTRY[name](cfg["embeddings"]) for name in active]
