from src.rag.retriever import Retriever
from src.rag.index import FAISSIndex
from src.embeddings import build_embedders
import yaml


def load_retriever(config_path: str = 'config.yaml') -> Retriever:
    """
    Single call to get a ready Retriever.
    ADK agents call this at startup.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rag_cfg = cfg['rag']
    variant = rag_cfg['embedding_variant']
    embs    = build_embedders(cfg)
    embedder = next(e for e in embs if e.name == variant)
    # todo: use config to determine which index class to instantiate (e.g. FAISS, Weaviate, etc)
    index = FAISSIndex(
        dim           = cfg['embeddings']['dim'],
        index_path    = rag_cfg['index_path'],
        metadata_path = rag_cfg['metadata_path'],
    )
    index.load()

    return Retriever(index, top_k=rag_cfg['top_k'])