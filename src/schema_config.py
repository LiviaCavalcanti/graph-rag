from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PathsConfig:
    """All system paths in one place."""

    joern_bin_dir: Path  # set via config `joern.bin_dir` / `paths.joern_bin_dir`
    output_dir: Path  # set via config `paths.output_dir` (default experiments/output)
    models_cache_dir: Path
    index_dir: Path


@dataclass
class GraphProcessingConfig:
    """All graph processing parameters (currently scattered in pipeline.py)."""

    slice_depth: int = 3
    change_weight: Dict[str, float] = field(
        default_factory=lambda: {
            "function_added": 1.0,
            "function_deleted": 1.0,
            "parameter_changed": 0.5,
        }
    )
    noise_types: List[str] = field(default_factory=lambda: ["add_noise", "drop_noise"])


@dataclass
class VariantConfig:
    """Variant registry (currently hard-coded in autopatch.py)."""

    name: str
    model: str
    llm_output_file: str  # was: "gpt-4_response.json" (hard-coded)
    patch_file: str


@dataclass
class EmbeddingConfig:
    """All embedding parameters (scattered defaults)."""

    variant: str
    dim: int = 128
    model_name: str = "codebert-base"
    model_checkpoint: Optional[Path] = None
    wl_iterations: int = 4
    wl_color_space: int = 8192
    hidden_dim: int = 64


@dataclass
class AgentConfig:
    """Patching-agent architecture + prompt selection (config ``agents:`` block).

    Typed accessor for the ``agents:`` section of ``config.yaml``. Read from a
    raw config dict via :meth:`from_cfg`; the batch runner and experiments vary
    ``architecture`` / ``prompt_variant`` as grid axes.
    """

    architecture: str = "single_turn"
    prompt_variant: str = "default"
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4096
    max_tool_iters: int = 6
    tools: Dict[str, Any] = field(default_factory=lambda: {"provider": "none"})
    multistep: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "AgentConfig":
        """Build from a full config dict, falling back to legacy locations."""
        raw = (cfg or {}).get("agents", {}) or {}
        return cls(
            architecture=raw.get("architecture", "single_turn"),
            prompt_variant=(
                raw.get("prompt_variant")
                or (cfg or {}).get("rag", {}).get("prompt_variant", "default")
            ),
            model=raw.get("model"),
            temperature=raw.get("temperature", 0.2),
            max_tokens=raw.get("max_tokens", 4096),
            max_tool_iters=raw.get("max_tool_iters", 6),
            tools=raw.get("tools", {"provider": "none"}),
            multistep=raw.get("multistep", {}),
        )

    def llm_params(self) -> dict:
        """Kwargs for ``AutoPatchPatcher`` / agent builders."""
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}


@dataclass
class AppConfig:
    """Root config, loaded from YAML."""

    paths: PathsConfig
    graph: GraphProcessingConfig
    embeddings: Dict[str, EmbeddingConfig]
    variants: List[VariantConfig]
    rag: Dict  # Your existing RAG config
    data: Dict  # Your existing data config
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    """Original YAML dict, kept for call sites that still need raw/legacy access."""

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "AppConfig":
        """Load ``path`` as YAML and build a typed :class:`AppConfig`."""
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict) -> "AppConfig":
        """Build a typed :class:`AppConfig` from a raw (legacy-shaped) config dict."""
        cls._validate(cfg)

        paths_raw = cfg.get("paths", {})
        joern_bin = (
            paths_raw.get("joern_bin_dir") or cfg.get("joern", {}).get("bin_dir") or ""
        )

        rag_cfg = cfg.get("rag", {})
        index_path = rag_cfg.get("index_path", "indexes/faiss.index")
        inferred_index_dir = str(Path(index_path).parent) if index_path else "indexes"

        paths_cfg = PathsConfig(
            joern_bin_dir=Path(joern_bin),
            output_dir=Path(paths_raw.get("output_dir", "experiments/output")),
            models_cache_dir=Path(paths_raw.get("models_cache_dir", "models")),
            index_dir=Path(paths_raw.get("index_dir", inferred_index_dir)),
        )

        graph_raw = cfg.get("graph_processing", {})
        graph_cfg = GraphProcessingConfig(
            slice_depth=graph_raw.get("slice_depth", 3),
            change_weight=graph_raw.get(
                "change_weight",
                {
                    "function_added": 1.0,
                    "function_deleted": 1.0,
                    "parameter_changed": 0.5,
                },
            ),
            noise_types=graph_raw.get("noise_types", ["add_noise", "drop_noise"]),
        )

        emb_root = cfg.get("embeddings", {})
        active_embedders = emb_root.get("active", [])
        if not active_embedders and cfg.get("rag", {}).get("embedding_variant"):
            active_embedders = [cfg["rag"]["embedding_variant"]]
            print(f"No active embeddings. Using {active_embedders} instead")

        embeddings_cfg: Dict[str, EmbeddingConfig] = {}
        for name in active_embedders:
            sub = (
                emb_root.get(name, {})
                if isinstance(emb_root.get(name, {}), dict)
                else {}
            )
            model_checkpoint = sub.get("checkpoint_path") or sub.get("model_checkpoint")
            model_name = sub.get("model_name") or sub.get("model_path") or name

            embeddings_cfg[name] = EmbeddingConfig(
                variant=name,
                dim=emb_root.get("dim", 128),
                model_name=str(model_name),
                model_checkpoint=Path(model_checkpoint) if model_checkpoint else None,
                wl_iterations=emb_root.get("wl", {}).get("num_iterations", 4),
                wl_color_space=emb_root.get("wl", {}).get("color_space", 8192),
                hidden_dim=sub.get(
                    "hidden_dim", emb_root.get("wl", {}).get("hidden_dim", 64)
                ),
            )

        variants_cfg: List[VariantConfig] = []
        for raw_variant in cfg.get("variants", []):
            if not isinstance(raw_variant, dict):
                continue
            name = raw_variant.get("name")
            model = raw_variant.get("model")
            if not name or not model:
                continue
            variants_cfg.append(
                VariantConfig(
                    name=name,
                    model=model,
                    llm_output_file=raw_variant.get(
                        "llm_output_file", f"{name}_response.json"
                    ),
                    patch_file=raw_variant.get("patch_file", f"{name}_patch.py"),
                )
            )

        return cls(
            paths=paths_cfg,
            graph=graph_cfg,
            embeddings=embeddings_cfg,
            variants=variants_cfg,
            rag=cfg.get("rag", {}),
            data=cfg.get("data", {}),
            raw=cfg,
        )

    @staticmethod
    def _validate(cfg: dict) -> None:
        """Fail fast on missing top-level config sections used by main modes."""
        required = ["data", "rag", "embeddings"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")

        if "joern" not in cfg and "paths" not in cfg:
            raise KeyError(
                "Missing joern/path configuration. Need cfg['joern'] or cfg['paths']"
            )


@dataclass
class DatasetBatch:
    """Contract emitted by the dataset stage.

    Represents one batch of query/index candidates plus provenance metadata.
    """

    batch_id: str
    run_id: str
    pairs: List[Any]
    split_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedBatch:
    """Contract emitted by the embedding stage.

    Each embedding row should align by index with `pairs`.
    """

    batch_id: str
    run_id: str
    embedder_name: str
    embedder_version: Optional[str]
    dim: int
    pairs: List[Any]
    embeddings: List[List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexUpdateResult:
    """Contract emitted by the indexing stage."""

    run_id: str
    index_backend: str
    index_path: Path
    index_version: str
    added_count: int
    total_count: int
    metadata_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Contract emitted by the retrieval stage for one query item."""

    run_id: str
    query_id: str
    query_cve: Optional[str]
    retriever_name: str
    top_k: int
    hit_ids: List[str]
    hit_scores: List[float]
    hit_metadata: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchResult:
    """Contract emitted by the patching stage for one query item."""

    run_id: str
    query_id: str
    patcher_name: str
    model_name: str
    prompt_version: Optional[str]
    patch_text: str
    raw_response: Optional[str] = None
    retrieval: Optional[RetrievalResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Contract emitted by the evaluation stage for one generated patch."""

    run_id: str
    query_id: str
    passed: bool
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
