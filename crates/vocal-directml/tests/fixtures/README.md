# DirectML retrieval fixtures

The two 299-byte FAISS IVF-Flat-L2 files share 2-dimensional centroids
`[(0,0), (10,0), (20,0), (30,0)]` and vectors
`[(0,1), (0,2), (10,1), (10,2), (30,1), (30,2)]`.
List 2 is empty; the remaining lists contain only two vectors. Their persisted
`nprobe` values are 1 and 2. These deliberately exercise empty candidate sets,
padding, fewer than eight neighbors, exact matches and multiple probes.

Recreate from this directory with Python FAISS:

```python
import faiss
import numpy as np

quantizer = faiss.IndexFlatL2(2)
quantizer.add(np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32))
index = faiss.IndexIVFFlat(quantizer, 2, 4, faiss.METRIC_L2)
index.is_trained = True
index.add(np.array([[0, 1], [0, 2], [10, 1], [10, 2], [30, 1], [30, 2]], dtype=np.float32))
for probes in (1, 2):
    index.nprobe = probes
    faiss.write_index(index, f"ivf-small-probe{probes}.index")
```
