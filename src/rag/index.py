import faiss
from pathlib import Path
from src.data.base import FunctionPair
import numpy as np
import json

class FAISSIndex:
    def __init__(self, dim: int, index_path:str, metadata_path:str):
        self.dim = dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add(self, pair:FunctionPair, embedding: np.ndarray, variant: str):
        vec = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        self.metadata.append({
            'cve_id':    pair.cve_id,
            'cwe_id':    pair.cwe_id,
            'func_name': pair.func_name,
            'project':   pair.project,
            'variant':   variant,
            'n_nodes':   pair.G_vuln.number_of_nodes(),
            **pair.meta,
        })

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2))
        print(f"Index saved {self.index.ntotal} vectors : {self.index_path}")
    
    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text())
        print(f"Index loaded: {self.index.ntotal} vectors")