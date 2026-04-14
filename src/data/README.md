# Adding a new dataset

1. Create `data/mydata.py` subclassing `BaseDataset`
2. Implement `name()`, `stream()` → yields `FunctionPair`, and `export_jobs()` → yields `ExportJob`
3. Register in `main.py`:
```python
from data.mydata import MyDataset
DATASETS = { ..., 'mydata': MyDataset }
```
4. Add config block in `config.yaml` under `data.mydata`

## Adding a new embedder

1. Create `embeddings/myembedder.py` subclassing `BaseEmbedder`
2. Implement `name` property and `embed_one(G) -> np.ndarray`
3. Register in `embeddings/__init__.py`:
```python
from .myembedder import MyEmbedder
REGISTRY = { ..., 'myembedder': MyEmbedder }
```
4. Add to `embeddings.active` list in `config.yaml`
