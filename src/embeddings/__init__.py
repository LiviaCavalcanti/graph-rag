from .netlsd import NetLSDEmbedder

REGISTRY: dict[str, type] = {
    'netlsd': NetLSDEmbedder,
}

def build_embedders(cfg: dict) -> list:
    active = cfg['embeddings']['active']
    return [REGISTRY[name](cfg['embeddings']) for name in active]