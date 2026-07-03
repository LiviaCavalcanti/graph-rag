from .codebert_seq import CodeBERTFlowEmbedder, CodeBERTSeqEmbedder
from .codexglue_baseline import CodeXGLUEBaselineEmbedder
from .combined import CombinedEmbedder, CombinedEnrichedEmbedder
from .gin import GINEmbedder, GINEnrichedEmbedder
from .gin_model import GINCodeBERTEmbedder
from .motif import MotifEmbedder
from .netlsd import NetLSDEmbedder
from .rgcn import RGCNEmbedder
from .vuln_pattern import (
    CodeBERTFlowPatternEmbedder,
    CodeBERTPatternEmbedder,
    VulnPatternEmbedder,
)
from .wl import WLEmbedder

REGISTRY: dict[str, type] = {
    "netlsd": NetLSDEmbedder,
    "wl": WLEmbedder,
    "gin": GINEmbedder,
    "gin_enriched": GINEnrichedEmbedder,
    "gin_codebert": GINCodeBERTEmbedder,
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
    unknown = [name for name in active if name not in REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown embedder(s) {unknown} in embeddings.active. "
            f"Available: {sorted(REGISTRY)}"
        )
    return [REGISTRY[name](cfg["embeddings"]) for name in active]
