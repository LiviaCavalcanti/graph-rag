from src.rag.hnsw import HNSWIndex
from pathlib import Path
import json
from cvefixes_experiments.scripts.pipeline_verification.exp_pipeline_verification import stratified_split
from src.metrics.retrieval_eval import (
    cve_retrieval_metrics,
    cwe_recall_metrics,
    retrieve_all,
)
from src.rag.retriever import Retriever
from src.embeddings import REGISTRY as EMBEDDER_REGISTRY

KS = [1, 5, 10]

SEED=42
# path to index
root_index_path = "cvefixes_experiments/output"

index = ["pipeline_verification_structure_file_ctxt2", "pipeline_verification_structure_file_ctxt"]

# Load pairs from cache
cache_path = Path(root_index_path) / index[0] / "cpg_cache"
with open(cache_path / "pairs.json", "r") as f:
    pairs = json.load(f)

# 3. Split
print(f"\n[3/5] Splitting into index/query (stratified by CWE)...")
index_pairs, query_pairs = stratified_split(pairs, test_ratio=0.2, seed=SEED)

for i in index:
    index_path = Path(root_index_path) / i / "indices"
    # get file names directly from the directory
    index_dir = index_path
    embedder = None
    retriever = Retriever(index)
    
    index_file_path = None
    metadata_path = None
    for file in index_dir.iterdir():
        if file.suffix == ".index":
            index_file_path = str(file)
        elif file.suffix == ".json":
            metadata_path = str(file)
    
    emb_name = index_file_path.split('__')[0]
    embedder = EMBEDDER_REGISTRY[emb_name]()
    # Load metadata to get dimension
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    dim = metadata.get("dimension", 384)  # default to 384 if not found
    
    index = HNSWIndex(dim,
                    index_path=index_file_path,
                    metadata_path=metadata_path)
    index.load()
    qr = retrieve_all(query_pairs, embedder, retriever, top_k=max(KS))
    print(f"    Retrieved for {len(qr)} queries")

    # Compute metrics
    cve_metrics = cve_retrieval_metrics(qr, ks=KS, index_metadata=index.metadata)
    cwe_metrics = cwe_recall_metrics(qr, index.metadata, top_k=max(KS))
